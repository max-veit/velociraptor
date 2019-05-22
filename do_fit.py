#!/usr/bin/env python
"""Script to automate the process of fitting a model"""


import argparse
import logging
import sys

import ase
import ase.io
import numpy as np

import fitutils
import transform

#TODO(max) remove when debugged
import pdb


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Fit a model for the given set of dipoles",
    epilog="Charges are assumed to sum to zero for each geometry, unless "
    "the geometries have a property (info entry, in ASE terminology) named "
    "'total_charge'.\n\n"
    "Setting either of the (scalar or tensor) weights to zero will turn "
    "off that component completely and the corresponding kernel file(s) "
    "will not be read.")
parser.add_argument('geometries', help="Geometries of the molecules in the "
    "fit; should be the name of a file readable by ASE.")
parser.add_argument('dipoles', help="Dipoles, in Cartesian coordinates, "
    "per geometry.  Entries must be in the same order as the geometry file.")
parser.add_argument('scalar_kernel_sparse', help="Filename for the "
    "sparse-sparse (MM) scalar kernel, in atomic environment space")
parser.add_argument('scalar_kernel', help="Filename for the full-sparse "
    "(NM) scalar kernel, in atomic environment space")
parser.add_argument('tensor_kernel_sparse', help="Filename for the "
    "sparse-sparse tensor kernel")
parser.add_argument('tensor_kernel', help="Filename for the "
    "full-sparse tensor kernel, mapping Cartesian components to environments")
parser.add_argument('weights_output', help="Name of a file into which to "
     "write the output weights")
parser.add_argument('-ws', '--scalar-weight', type=float, metavar='weight',
    help="Weight of the scalar component (charges) in the model",
    required=True)
parser.add_argument('-wt', '--tensor-weight', type=float, metavar='weight',
    help="Weight of the tensor component (point dipoles) in the model",
    required=True)
parser.add_argument('-rc', '--charge-regularization', type=float, default=1.0,
                    metavar='sigma_q', help="Regularization coefficient "
                    "(sigma) for total charges")
parser.add_argument('-rd', '--dipole-regularization', type=float,
                    required=True, metavar='sigma_mu', help="Regularization "
                    "coefficient (sigma) for dipole components")
parser.add_argument('-nt', '--num-training-geometries', type=int,
                    metavar='<n>', default=-1,
                    help="Keep only the first <n> geometries for training.")
parser.add_argument('-sj', '--sparse-jitter', type=float, default=1E-9,
                    help="Small positive constant to ensure positive "
                    "definiteness of the kernel matrix")
parser.add_argument('-m', '--charge-mode', choices=['none', 'fit', 'lagrange'],
                    help="How to control the total charge of each geometry. "
                    "Choices are 'none' (just fit dipoles), 'fit', (fit "
                    "dipoles and charges), and 'lagrange' (constrain total "
                    "charges exactly using Lagrange multipliers).",
                    default='fit')
parser.add_argument('-pr', '--print-residuals', action='store_true',
                    help="Print the RMSE residuals of the model evaluated on "
                    "its own training data")
parser.add_argument('-wr', '--write-residuals', metavar='FILE',
                    help="File in which to write the individual (non-RMSed) "
                    "residuals.  If not given, these will not be written.")
parser.add_argument('-pw', '--print-weight-norm', action='store_true',
                    help="Print the norm of the weights? (useful for quick "
                    "sanity checks)")
parser.add_argument('-mm', '--memory-map', action='store_true',
                    help="Memory-map the larger kernels to save memory? (they "
                    "will still be read in after slicing and transforming)")


def load_kernels(args):
    """Load the kernels from files"""
    if args.memory_map:
        mmap_mode = 'r'
    else:
        mmap_mode = None
    if args.scalar_weight != 0:
        scalar_kernel_sparse = np.load(args.scalar_kernel_sparse)
        scalar_kernel = np.load(args.scalar_kernel, mmap_mode=mmap_mode)
    else:
        scalar_kernel = np.array([])
        scalar_kernel_sparse = np.array([])
    if args.tensor_weight != 0:
        tensor_kernel_sparse = np.load(args.tensor_kernel_sparse)
        tensor_kernel = np.load(args.tensor_kernel, mmap_mode=mmap_mode)
    else:
        tensor_kernel = np.array([])
        tensor_kernel_sparse = np.array([])
    return (scalar_kernel_sparse, scalar_kernel,
            tensor_kernel_sparse, tensor_kernel)


def get_charges(geometries):
    return np.array([geom.info.get('total_charge', 0.) for geom in geometries])


def transform_kernels(geometries, scalar_kernel_full_sparse, scalar_weight,
                                  tensor_kernel_full_sparse, tensor_weight):
    if scalar_weight != 0:
        scalar_kernel_transformed = transform.transform_envts_charge_dipoles(
                geometries, scalar_kernel_full_sparse)
        scalar_kernel_transformed *= scalar_weight
    else:
        scalar_kernel_transformed = scalar_kernel_full_sparse
    # Assuming the spherical-to-Cartesian transformation was done elsewhere
    if tensor_weight != 0:
        tensor_kernel_transformed = (
                transform.transform_vector_envts_charge_dipoles(
                    geometries, tensor_kernel_full_sparse))
        tensor_kernel_transformed *= tensor_weight

    else:
        tensor_kernel_transformed = tensor_kernel_full_sparse
    return scalar_kernel_transformed, tensor_kernel_transformed


def transform_sparse_kernels(geometries, scalar_kernel_sparse, scalar_weight,
                                         tensor_kernel_sparse, tensor_weight):
    if scalar_weight != 0:
        scalar_kernel_sparse = scalar_kernel_sparse * scalar_weight
    if tensor_weight != 0:
        kernel_shape = tensor_kernel_sparse.shape
        if kernel_shape[2:] != (3, 3) or kernel_shape[0] != kernel_shape[1]:
            raise ValueError('Vector kernel has unrecognized shape: {}, was'
                    'expecting something of the form (n_sparse, n_sparse, '
                    '3, 3)'.format(kernel_shape))
        tensor_kernel_sparse = (
                tensor_kernel_sparse.transpose((0, 2, 1, 3)).reshape(
                    (kernel_shape[0]*3, kernel_shape[1]*3))
                * tensor_weight)
    return scalar_kernel_sparse, tensor_kernel_sparse


def compute_weights(args, dipoles, charges,
                    scalar_kernel_sparse, scalar_kernel_transformed,
                    tensor_kernel_sparse, tensor_kernel_transformed):
    if (args.scalar_weight == 0) and (args.tensor_weight == 0):
        raise ValueError("Both weights set to zero, can't fit with no data.")
    elif ((args.charge_mode != 'fit') and (args.scalar_weight != 0)
                                      and (args.tensor_weight != 0)):
        raise ValueError("Combined fitting only works with 'fit' charge-mode")
    if args.charge_mode == 'fit' and args.scalar_weight == 0:
        args.charge_mode = 'none'
        logger.warning("Doing tensor kernel fitting with charges; since l=1 "
                       "kernels are insensitive to scalars, this is exactly "
                       "the same as tensor fitting without charges.")
    if args.charge_mode == 'none':
        regularizer = args.dipole_regularization**-2 * np.ones((dipoles.size,))
    else:
        regularizer = fitutils.make_inv_reg_vector(args.dipole_regularization,
                                                   args.charge_regularization,
                                                   len(charges))
    if args.charge_mode == 'none':
        if args.tensor_weight == 0:
            scalar_kernel_transformed = np.delete(
                    scalar_kernel_transformed, slice(None, None, 4), axis=0)
            return fitutils.compute_weights(
                    dipoles, scalar_kernel_sparse,
                    scalar_kernel_transformed, regularizer,
                    args.sparse_jitter)
        elif args.scalar_weight == 0:
            tensor_kernel_transformed = np.delete(
                    tensor_kernel_transformed, slice(None, None, 4), axis=0)
            return fitutils.compute_weights(
                    dipoles, tensor_kernel_sparse,
                    tensor_kernel_transformed, regularizer,
                    args.sparse_jitter)
        else:
            raise ValueError("Can't do combined fitting without charges")
    elif args.charge_mode == 'fit':
        if args.tensor_weight == 0:
            return fitutils.compute_weights_charges(
                    charges, dipoles,
                    scalar_kernel_sparse, scalar_kernel_transformed,
                    regularizer, args.sparse_jitter)
        else:
            return fitutils.compute_weights_two_model(
                        charges, dipoles,
                        scalar_kernel_sparse, scalar_kernel_transformed,
                        tensor_kernel_sparse, tensor_kernel_transformed,
                        regularizer, args.sparse_jitter)
    elif args.charge_mode == 'lagrange':
        if args.tensor_weight != 0:
            raise ValueError("Charge constraints not yet implemented together "
                             "with tensor fitting")
        weights = fitutils.compute_weights_charge_constrained(
                charges, dipoles,
                scalar_kernel_sparse, scalar_kernel_transformed, regularizer,
                args.sparse_jitter)
        return weights
    else:
        raise ValueError("Unrecognized charge mode '%s'".format(charge_mode))


def compute_own_residuals(
        args, weights, dipoles, charges, natoms_list,
        scalar_kernel_transformed, tensor_kernel_transformed):
    if args.charge_mode != 'none' and args.scalar_weight != 0:
        charges_test = charges
    else:
        charges_test = None
        scalar_kernel_transformed = np.delete(
                scalar_kernel_transformed, slice(None, None, 4), axis=0)
        tensor_kernel_transformed = np.delete(
                tensor_kernel_transformed, slice(None, None, 4), axis=0)
    if args.tensor_weight == 0:
        kernels = scalar_kernel_transformed
    elif args.scalar_weight == 0:
        kernels = tensor_kernel_transformed
    else:
        kernels = [scalar_kernel_transformed, tensor_kernel_transformed]
        weights = np.split(weights,
                           np.array([scalar_kernel_transformed.shape[1]]))
    intrinsic_dipole_std = vars(args).get('intrinsic_variation')
    residuals = fitutils.compute_residuals(
        weights, kernels, dipoles, natoms_list,
        charges_test=charges_test, return_rmse=args.print_residuals,
        intrinsic_dipole_std=intrinsic_dipole_std)
    if 'dipole_rmse' in residuals:
        print("Dipole RMSE: {:.10f} : {:.10f} of intrinsic variation".format(
            residuals['dipole_rmse'], residuals['dipole_frac']))
    if 'charge_rmse' in residuals:
        print("Charge RMSE: {:.10f} : {:.10f} of intrinsic variation".format(
            residuals['charge_rmse'], residuals['charge_frac']))
    if args.write_residuals is not None:
        np.savez(args.write_residuals, **residuals)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.num_training_geometries > 0:
        n_train = args.num_training_geometries
        geometries = ase.io.read(args.geometries, slice(0, n_train))
    else:
        geometries = ase.io.read(args.geometries, slice(None))
        n_train = len(geometries)
    charges = get_charges(geometries)
    dipoles = np.loadtxt(args.dipoles)[:n_train]
    natoms_list = [geom.get_number_of_atoms() for geom in geometries]
    n_descriptors = sum(natoms_list)
    (scalar_kernel_sparse, scalar_kernel_full_sparse,
     tensor_kernel_sparse, tensor_kernel_full_sparse) = load_kernels(args)
    scalar_kernel_full_sparse = scalar_kernel_full_sparse[:n_descriptors]
    tensor_kernel_full_sparse = tensor_kernel_full_sparse[:n_descriptors]
    scalar_kernel_sparse, tensor_kernel_sparse = transform_sparse_kernels(
        geometries, scalar_kernel_sparse, args.scalar_weight,
                    tensor_kernel_sparse, args.tensor_weight)
    scalar_kernel_transformed, tensor_kernel_transformed = transform_kernels(
        geometries, scalar_kernel_full_sparse, args.scalar_weight,
                    tensor_kernel_full_sparse, args.tensor_weight)
    # Close files or free memory for what comes next
    del scalar_kernel_full_sparse
    del tensor_kernel_full_sparse
    weights = compute_weights(
        args, dipoles, charges,
        scalar_kernel_sparse, scalar_kernel_transformed,
        tensor_kernel_sparse, tensor_kernel_transformed)
    np.save(args.weights_output, weights)
    if args.print_residuals or (args.write_residuals is not None):
        compute_own_residuals(args, weights, dipoles, charges, natoms_list,
                              scalar_kernel_transformed,
                              tensor_kernel_transformed)
    if args.print_weight_norm:
        print("Norm (L2) of weights: {:.4f}".format(np.linalg.norm(weights)))

