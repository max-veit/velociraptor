#!/usr/bin/env python
"""Script to automate the process of fitting a model"""


import argparse
import logging
import os

import ase.io
import numpy as np

from velociraptor.fitutils import (transform_kernels, transform_sparse_kernels,
                                   compute_weights, compute_residuals, get_charges)


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Fit a model for the given set of dipoles",
    epilog="Charges are assumed to sum to zero for each geometry, unless "
    "the geometries have a property (info entry, in ASE terminology) named "
    "'total_charge'.\n\n"
    "Setting either of the (scalar or vector) weights to zero will turn "
    "off that component completely and the corresponding kernel file(s) "
    "will not be read.")
parser.add_argument(
    'geometries', help="Geometries of the molecules in the fit; should be the "
            "name of a file readable by ASE.")
parser.add_argument(
    'dipoles', help="Dipoles, in Cartesian coordinates, per geometry.  "
            "Entries must be in the same order as the geometry file.")
parser.add_argument(
    'scalar_kernel_sparse', help="Filename for the sparse-sparse (MM) scalar "
            "kernel, in atomic environment space")
parser.add_argument(
    'scalar_kernel', help="Filename for the full-sparse (NM) scalar kernel, "
            "in atomic environment space")
parser.add_argument(
    'vector_kernel_sparse', help="Filename for the sparse-sparse "
            "vector kernel")
parser.add_argument(
    'vector_kernel', help="Filename for the full-sparse vector kernel, "
            "mapping Cartesian components to environments")
parser.add_argument(
    'weights_output', help="Name of a file into which to write the "
            "output weights")
parser.add_argument(
    '-ws', '--scalar-weight', type=float, metavar='weight',
            help="Weight of the scalar component (charges) in the model",
            required=True)
parser.add_argument(
    '-wt', '--vector-weight', type=float, metavar='weight',
            help="Weight of the vector component (point dipoles) in the model",
            required=True)
parser.add_argument(
    '-rc', '--charge-regularization', type=float, default=1.0,
            metavar='sigma_q', help="Regularization coefficient (sigma) "
            "for total charges")
parser.add_argument(
    '-rd', '--dipole-regularization', type=float, required=True,
            metavar='sigma_mu', help="Regularization coefficient (sigma) "
            "for dipole components")
parser.add_argument(
    '-dn', '--dipole-no-normalize',
            action='store_false', dest='dipole_normalize',
            help="Don't normalize the dipole by the number of atoms before "
            "fitting (the fit is usally better _with_ normalization)")
parser.add_argument(
    '-nt', '--num-training-geometries', type=int, metavar='<n>', default=-1,
            help="Keep only the first <n> geometries for training.")
parser.add_argument(
    '-sj', '--sparse-jitter', type=float, default=0.0, help="Small positive "
            "constant to ensure positive definiteness of the kernel matrix"
            " (Warning: Deprecated in favour of np.lstsq to strip out small "
            " eigenvalues in a more systematic way)")
parser.add_argument(
    '-m', '--charge-mode', choices=['none', 'fit', 'lagrange'],
            help="How to control the total charge of each geometry. Choices "
            "are 'none' (just fit dipoles), 'fit', (fit dipoles and charges), "
            "and 'lagrange' (constrain total charges exactly using "
            "Lagrange multipliers).", default='fit')
parser.add_argument(
    '-pr', '--print-residuals', action='store_true', help="Print the RMSE "
            "residuals of the model evaluated on its own training data")
parser.add_argument(
    '-wr', '--write-residuals', metavar='FILE', help="File in which to write "
            "the individual (non-RMSed) residuals.  If not given, these will "
            "not be written.")
parser.add_argument(
    '-pw', '--print-weight-norm', action='store_true', help="Print the norm "
            "of the weights? (useful for quick sanity checks)")
parser.add_argument(
    '-pc', '--print-condition-number', action='store_true', help="Print the "
            "condition number of the linear system to be solved?")
parser.add_argument(
    '-cc', '--condition-cutoff', type=float, metavar='rcond',
            help="Condition-number cutoff to "
            "use for the least-squares solver np.linalg.lstsq().  The default "
            "(the \"new\" default of lstsq) should be sensible; otherwise try "
            "the previous default of 10^-15 (the difference should be small "
            "in practice).")
parser.add_argument(
    '-mm', '--memory-map', action='store_true', help="Memory-map the larger "
            "kernels to save memory? (they will still be read in after "
            "slicing and transforming)")
parser.add_argument(
    '-tk', '--transpose-full-kernels', action='store_true', help="Transpose "
            "the full-sparse kernels, assuming they were stored in the "
            "opposite order (MN) from the one expected (NM) (where N is full "
            "and M is sparse)")
parser.add_argument(
    '-tm', '--vector-kernel-molecular', action='store_true', help="Is the full"
            " vector kernel stored in molecular, rather than atomic, format? "
            "(i.e. are they pre-summed over the atoms in a molecule?) "
            "Note this option is compatible with -tk and -tvk.")
parser.add_argument(
    '-st', '--spherical-tensor-ordering', action='store_true',
           dest='spherical', help="Transform the vector kernels from spherical"
           " tensor to the internal Cartesian ordering")


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
    if args.vector_weight != 0:
        vector_kernel_sparse = np.load(args.vector_kernel_sparse)
        vector_kernel = np.load(args.vector_kernel, mmap_mode=mmap_mode)
    else:
        vector_kernel = np.array([])
        vector_kernel_sparse = np.array([])
    del args.scalar_kernel_sparse
    del args.vector_kernel_sparse
    return (scalar_kernel_sparse, scalar_kernel,
            vector_kernel_sparse, vector_kernel)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.num_training_geometries > 0:
        n_train = args.num_training_geometries
        geometries = ase.io.read(args.geometries, slice(0, n_train))
    else:
        geometries = ase.io.read(args.geometries, slice(None))
        n_train = len(geometries)
    charges = get_charges(geometries)
    dipole_fext = os.path.splitext(args.dipoles)[1]
    if dipole_fext == '.npy':
        dipoles = np.load(args.dipoles)[:n_train]
    elif (dipole_fext == '.txt') or (dipole_fext == '.dat'):
        dipoles = np.loadtxt(args.dipoles)[:n_train]
    else:
        logger.warn("Dipoles file has no filename extension; assuming "
                    "plain text.")
        dipoles = np.loadtxt(args.dipoles)[:n_train]
    del args.dipoles
    natoms_list = [geom.get_number_of_atoms() for geom in geometries]
    if args.dipole_normalize:
        dipoles = (dipoles.T / natoms_list).T
        charges = charges / natoms_list
    n_descriptors = sum(natoms_list)
    (scalar_kernel_sparse, scalar_kernel_full_sparse,
     vector_kernel_sparse, vector_kernel_full_sparse) = load_kernels(args)
    #TODO move some of this logic into the transform functions?
    if args.vector_kernel_molecular:
        n_vector = n_train
    else:
        n_vector = n_descriptors
    if not args.transpose_full_kernels:
        vector_kernel_full_sparse = vector_kernel_full_sparse[:n_vector]
        scalar_kernel_full_sparse = scalar_kernel_full_sparse[:n_descriptors]
    else:
        vector_kernel_full_sparse = vector_kernel_full_sparse[:,:n_vector]
        scalar_kernel_full_sparse = scalar_kernel_full_sparse[:,:n_descriptors]
    scalar_kernel_sparse, vector_kernel_sparse = transform_sparse_kernels(
        geometries, scalar_kernel_sparse, args.scalar_weight,
                    vector_kernel_sparse, args.vector_weight, args.spherical)
    scalar_kernel_transformed, vector_kernel_transformed = transform_kernels(
        geometries, scalar_kernel_full_sparse, args.scalar_weight,
        vector_kernel_full_sparse, args.vector_weight,
        args.vector_kernel_molecular, args.transpose_full_kernels,
        args.dipole_normalize, args.spherical)
    # Close files or free memory for what comes next
    del scalar_kernel_full_sparse
    del vector_kernel_full_sparse
    weights = compute_weights(
        dipoles, charges,
        scalar_kernel_sparse, scalar_kernel_transformed,
        vector_kernel_sparse, vector_kernel_transformed,
        **vars(args))
    np.save(args.weights_output, weights)
    if args.print_residuals or (args.write_residuals is not None):
        args.dipole_normalized = args.dipole_normalize
        compute_residuals(weights, dipoles, charges, natoms_list,
                          scalar_kernel_transformed,
                          vector_kernel_transformed, **vars(args))
    if args.print_weight_norm:
        print("Norm (L2) of weights: {:.4f}".format(np.linalg.norm(weights)))

