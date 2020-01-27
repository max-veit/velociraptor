"""High-level utilities for fitting, preparation, and post-processing"""


import logging
import numpy as np

from . import solvers
from .transform import (transform_envts_charge_dipoles,
                        transform_envts_partial_charges,
                        transform_vector_envts_charge_dipoles,
                        transform_vector_envts_atomic_dipoles,
                        transform_vector_mols_charge_dipoles,
                        transform_spherical_tensor_to_cartesian)

logger = logging.getLogger(__name__)


def get_charges(geometries):
    return np.array([geom.info.get('total_charge', 0.) for geom in geometries])


def transform_kernels(geometries, scalar_kernel_full_sparse, scalar_weight,
                      vector_kernel_full_sparse, vector_weight,
                      vector_kernel_molecular=False,
                      transpose_scalar_kernel=False,
                      transpose_vector_kernel=False,
                      dipole_normalize=True, spherical=False):
    if scalar_weight != 0:
        scalar_kernel_transformed = transform_envts_charge_dipoles(
                geometries, scalar_kernel_full_sparse, transpose_scalar_kernel,
                dipole_normalize)
        scalar_kernel_transformed *= scalar_weight
    else:
        scalar_kernel_transformed = scalar_kernel_full_sparse
    # Assuming the spherical-to-Cartesian transformation was done elsewhere
    if vector_weight != 0:
        if vector_kernel_molecular:
            vector_kernel_transformed = transform_vector_mols_charge_dipoles(
                        geometries, vector_kernel_full_sparse,
                        transpose_vector_kernel, (not dipole_normalize),
                        spherical)
        else:
            vector_kernel_transformed = transform_vector_envts_charge_dipoles(
                        geometries, vector_kernel_full_sparse,
                        transpose_vector_kernel, dipole_normalize, spherical)
        vector_kernel_transformed *= vector_weight
    else:
        vector_kernel_transformed = vector_kernel_full_sparse
    return scalar_kernel_transformed, vector_kernel_transformed


def transform_sparse_kernels(geometries, scalar_kernel_sparse, scalar_weight,
                                         tensor_kernel_sparse, tensor_weight,
                                         spherical=False):
    if scalar_weight != 0:
        scalar_kernel_sparse = scalar_kernel_sparse * scalar_weight
    if tensor_weight != 0:
        kernel_shape = tensor_kernel_sparse.shape
        if kernel_shape[2:] != (3, 3) or kernel_shape[0] != kernel_shape[1]:
            raise ValueError(
                    "Vector kernel has unrecognized shape: {}, was"
                    "expecting something of the form (n_sparse, n_sparse, "
                    "3, 3)".format(kernel_shape))
        if spherical:
            tensor_kernel_sparse = transform_spherical_tensor_to_cartesian(
                    tensor_kernel_sparse)
        tensor_kernel_sparse = (
                tensor_kernel_sparse.transpose((0, 2, 1, 3)).reshape(
                    (kernel_shape[0]*3, kernel_shape[1]*3))
                * tensor_weight)
    return scalar_kernel_sparse, tensor_kernel_sparse


def compute_weights(dipoles, charges,
                    scalar_kernel_sparse, scalar_kernel_transformed,
                    tensor_kernel_sparse, tensor_kernel_transformed,
                    scalar_weight=0, tensor_weight=0, charge_mode='none',
                    dipole_regularization=1.0, charge_regularization=1.0,
                    sparse_jitter=1E-10, print_condition_number=False,
                    condition_cutoff=None, **extra_args):
    if (scalar_weight == 0) and (tensor_weight == 0):
        raise ValueError("Both weights set to zero, can't fit with no data.")
    elif ((charge_mode != 'fit') and (scalar_weight != 0)
                                 and (tensor_weight != 0)):
        raise ValueError("Combined fitting only works with 'fit' charge-mode")
    if charge_mode == 'fit' and scalar_weight == 0:
        charge_mode = 'none'
        logger.warning("Requested tensor kernel fitting with charges; since "
                       "l=1 kernels are insensitive to scalars, this is "
                       "exactly the same as tensor fitting without charges.")
    if charge_mode == 'none':
        regularizer = dipole_regularization**-2 * np.ones((dipoles.size,))
    else:
        regularizer = solvers.make_inv_reg_vector(dipole_regularization,
                                                  charge_regularization,
                                                  len(charges))
    if charge_mode == 'none':
        if tensor_weight == 0:
            scalar_kernel_transformed = np.delete(
                    scalar_kernel_transformed, slice(None, None, 4), axis=0)
            return solvers.compute_weights(
                    dipoles, scalar_kernel_sparse,
                    scalar_kernel_transformed, regularizer,
                    sparse_jitter,
                    print_condition=print_condition_number,
                    condition_cutoff=condition_cutoff)
        elif scalar_weight == 0:
            tensor_kernel_transformed = np.delete(
                    tensor_kernel_transformed, slice(None, None, 4), axis=0)
            return solvers.compute_weights(
                    dipoles, tensor_kernel_sparse,
                    tensor_kernel_transformed, regularizer,
                    sparse_jitter,
                    print_condition=print_condition_number,
                    condition_cutoff=condition_cutoff)
        else:
            raise ValueError("Can't do combined fitting without charges")
    elif charge_mode == 'fit':
        if tensor_weight == 0:
            return solvers.compute_weights_charges(
                    charges, dipoles,
                    scalar_kernel_sparse, scalar_kernel_transformed,
                    regularizer, sparse_jitter,
                    print_condition=print_condition_number,
                    condition_cutoff=condition_cutoff)
        else:
            return solvers.compute_weights_two_model(
                        charges, dipoles,
                        scalar_kernel_sparse, scalar_kernel_transformed,
                        tensor_kernel_sparse, tensor_kernel_transformed,
                        regularizer, sparse_jitter,
                        print_condition=print_condition_number,
                        condition_cutoff=condition_cutoff)
    elif charge_mode == 'lagrange':
        if tensor_weight != 0:
            raise ValueError("Charge constraints not yet implemented together "
                             "with tensor fitting")
        weights = solvers.compute_weights_charge_constrained(
                charges, dipoles,
                scalar_kernel_sparse, scalar_kernel_transformed, regularizer,
                sparse_jitter, condition_cutoff=condition_cutoff)
        return weights
    else:
        raise ValueError("Unrecognized charge mode '{:s}'".format(charge_mode))


def compute_residuals(
        weights, dipoles, charges, natoms_list,
        scalar_kernel_transformed, tensor_kernel_transformed,
        scalar_weight=0, tensor_weight=0, charge_mode=None,
        intrinsic_variation=None, print_residuals=True, write_residuals=None,
        dipole_normalized=False, **extra_args):
    if charge_mode != 'none' and scalar_weight != 0:
        charges_test = charges
    else:
        charges_test = None
        scalar_kernel_transformed = np.delete(
                scalar_kernel_transformed, slice(None, None, 4), axis=0)
        tensor_kernel_transformed = np.delete(
                tensor_kernel_transformed, slice(None, None, 4), axis=0)
    if tensor_weight == 0:
        kernels = scalar_kernel_transformed
    elif scalar_weight == 0:
        kernels = tensor_kernel_transformed
    else:
        kernels = [scalar_kernel_transformed, tensor_kernel_transformed]
        weights = np.split(weights,
                           np.array([scalar_kernel_transformed.shape[1]]))
    residuals = solvers.compute_residuals(
        weights, kernels, dipoles, natoms_list,
        charges_test=charges_test, return_rmse=True,
        intrinsic_dipole_std=intrinsic_variation,
        dipole_normalized=dipole_normalized)
    if print_residuals:
        if 'dipole_rmse' in residuals:
            print(
                "Dipole RMSE: {:.10f} : {:.10f} of intrinsic variation".format(
                    residuals['dipole_rmse'], residuals['dipole_frac']))
        if 'charge_rmse' in residuals:
            print(
                "Charge RMSE: {:.10f} : {:.10f} of intrinsic variation".format(
                    residuals['charge_rmse'], residuals['charge_frac']))
    if write_residuals is not None:
        np.savez(write_residuals, **residuals)
    return residuals


def compute_per_atom_properties(
        geometries, weights,
        scalar_kernel, tensor_kernel,
        scalar_weight=0, tensor_weight=0, transpose_kernels=False,
        spherical=False, write_properties=None, write_properties_geoms=None):
    if scalar_weight != 0:
        scalar_kernel_transformed = transform_envts_partial_charges(
                geometries, scalar_kernel, transpose_kernels) * scalar_weight
        n_scalar_envs = scalar_kernel_transformed.shape[0]
    if tensor_weight != 0:
        tensor_kernel_transformed = transform_vector_envts_atomic_dipoles(
                geometries, tensor_kernel,
                transpose_kernels, spherical) * tensor_weight
    per_atom_properties = dict()
    if tensor_weight == 0:
        per_atom_properties['charges'] = scalar_kernel_transformed.dot(weights)
    elif scalar_weight == 0:
        per_atom_properties['dipoles'] = tensor_kernel_transformed.dot(weights)
    else:
        per_atom_properties['charges'] = scalar_kernel_transformed.dot(
                weights[:n_scalar_envs])
        per_atom_properties['dipoles'] = tensor_kernel_transformed.dot(
                weights[n_scalar_envs:])
    if write_properties is not None:
        np.savez(write_properties, **per_atom_properties)
    if write_properties_geoms is not None:
        atom_index = 0
        for geom in geometries:
            n_atoms = len(geom)
            for key, prop in per_atom_properties.items():
                geom.arrays[key] = prop[atom_index : atom_index + n_atoms]
            atom_index += n_atoms

