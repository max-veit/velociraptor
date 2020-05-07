"""High-level utilities for fitting, preparation, and post-processing

Public functions:
    get_charges         Extract charges from an ASE Atoms list
    get_dipoles         Extract dipoles from an ASE Atoms list
    transform_kernels   Transform full-sparse kernels for fitting
    transform_sparse_kernels
                        Transform sparse-sparse kernels for fitting
    compute_weights     Compute the weights for a model (i.e. do a fit)
    compute_residuals   Compute the test residuals for a fitted model
    compute_per_atom_properties
                        Return per-atom predictions for a fitted model
"""


import logging

import ase.io
import numpy as np

from . import solvers
from .transform import (transform_envts_charge_dipoles,
                        transform_envts_partial_charges,
                        transform_vector_envts_charge_dipoles,
                        transform_vector_envts_atomic_dipoles,
                        transform_vector_mols_charge_dipoles,
                        transform_spherical_tensor_to_cartesian)

logger = logging.getLogger(__name__)


def get_charges(geometries, prop_name='total_charge'):
    """"Extract the total charge property from the ASE geometries

    The property name is given by the 'prop_name' argument,
    'total_charge' by default.

    Defaults to zero without throwing an error if the property is not
    present
    """
    return np.array([geom.info.get(prop_name, 0.) for geom in geometries])


def get_dipoles(geometries, prop_name='dipole'):
    """Extract the dipoles array from the ASE geometries

    The property name is given by 'prop_name', 'dipole' by default, and
    assumed to reside in the Atoms's 'arrays' dict.  Throws a KeyError
    if the property is missing from any one of the geometries.
    """
    return np.array([geom.arrays[prop_name] for geom in geometries])


def transform_kernels(geometries, scalar_kernel_full_sparse, scalar_weight,
                      vector_kernel_full_sparse, vector_weight,
                      vector_kernel_molecular=False,
                      transpose_full_kernels=False,
                      dipole_normalize=True, spherical=False):
    """Transform the full-sparse (NM) kernels for fitting

    Specifically, transforms from atom-sparsepoint (or spherical
    molecule-sparsepoint, available for vector kernels) to Cartesian
    dipole-sparsepoint kernels (with the first row for each molecule
    being total charge, the next three being x, y, and z dipole
    components).  Additionally applies overall scalar and vector weights
    (deltas).

    Parameters:
        geometries      List of ASE-compatible Atoms objects
        scalar_kernel_full_sparse
                        Full-sparse (NM) atomic scalar kernel
        scalar_weight   Overall scaling (delta) for scalar component
        vector_kernel_full_sparse
                        Full-sparse (NM) atomic or molecular vector
                        kernel
        vector_weight   Overall scaling (delta) for vector component
        vector_kernel_molecular
                        Whether the vector kernel is stored in the more
                        compact, pre-summed molecular (rather than
                        atomic) form
        transpose_full_kernels
                        Whether the kernels are transposed -- i.e.
                        sparse-full (MN) -- from the usual expected
                        ordering
        spherical       Whether the vector kernel is in spherical tensor
                        ordering (m = -1, 0, 1), rather than Cartesian
        dipole_normalize
                        Whether to make the kernels consistent with
                        dipole moments normalized by the number of atoms
                        (note that in the case of molecular vector
                        kernels, it is assumed that they will already be
                        normalized by the number of atoms -- setting
                        this to False will therefore undo this
                        normalization)

    Return the transformed scalar and vector kernels as a tuple
    """
    if scalar_weight != 0:
        scalar_kernel_transformed = transform_envts_charge_dipoles(
                geometries, scalar_kernel_full_sparse, transpose_full_kernels,
                dipole_normalize)
        scalar_kernel_transformed *= scalar_weight
    else:
        scalar_kernel_transformed = scalar_kernel_full_sparse
    # Assuming the spherical-to-Cartesian transformation was done elsewhere
    if vector_weight != 0:
        if vector_kernel_molecular:
            vector_kernel_transformed = transform_vector_mols_charge_dipoles(
                        geometries, vector_kernel_full_sparse,
                        transpose_full_kernels, (not dipole_normalize),
                        spherical)
        else:
            vector_kernel_transformed = transform_vector_envts_charge_dipoles(
                        geometries, vector_kernel_full_sparse,
                        transpose_full_kernels, dipole_normalize, spherical)
        vector_kernel_transformed *= vector_weight
    else:
        vector_kernel_transformed = vector_kernel_full_sparse
    return scalar_kernel_transformed, vector_kernel_transformed


def transform_sparse_kernels(geometries, scalar_kernel_sparse, scalar_weight,
                                         vector_kernel_sparse, vector_weight,
                                         spherical=False):
    """Transform the sparse-sparse (MM) kernels for fitting

    For the scalar kernel, this only performs the delta-scaling needed
    to make it consistent with the full-sparse kernels.  For the vector
    kernel, it additionally reorders the components if it was stored in
    spherical tensor ordering.

    Parameters:
        geometries      List of ASE-compatible Atoms objects
        scalar_kernel_sparse
                        Sparse-sparse (MM) atomic scalar kernel
        scalar_weight   Overall scaling (delta) for scalar component
        vector_kernel_sparse
                        Sparse-sparse (MM) atomic vector kernel
        vector_weight   Overall scaling (delta) for vector component
        spherical       Whether the vector kernel is in spherical tensor
                        ordering (m = -1, 0, 1), rather than Cartesian

    Return the transformed sparse scalar and vector kernels as a tuple
    """
    if scalar_weight != 0:
        scalar_kernel_sparse = scalar_kernel_sparse * scalar_weight
    if vector_weight != 0:
        kernel_shape = vector_kernel_sparse.shape
        if kernel_shape[2:] != (3, 3) or kernel_shape[0] != kernel_shape[1]:
            raise ValueError(
                    "Vector kernel has unrecognized shape: {}, was"
                    "expecting something of the form (n_sparse, n_sparse, "
                    "3, 3)".format(kernel_shape))
        if spherical:
            vector_kernel_sparse = transform_spherical_tensor_to_cartesian(
                    vector_kernel_sparse)
        vector_kernel_sparse = (
                vector_kernel_sparse.transpose((0, 2, 1, 3)).reshape(
                    (kernel_shape[0]*3, kernel_shape[1]*3))
                * vector_weight)
    return scalar_kernel_sparse, vector_kernel_sparse


def compute_weights(dipoles, charges,
                    scalar_kernel_sparse, scalar_kernel_transformed,
                    vector_kernel_sparse, vector_kernel_transformed,
                    scalar_weight=0, vector_weight=0, charge_mode='none',
                    dipole_regularization=1.0, charge_regularization=1.0,
                    sparse_jitter=1E-10, print_condition_number=False,
                    condition_cutoff=None, **extra_args):
    """Compute the fitting weights for scalar, vector, or combined models

    Parameters:
        dipoles         Training dipole moments to fit.  Normalization
                        _must_ be consistent with the 'dipole_normalize'
                        option given to the 'transform_kernels()'
                        function -- if True (the default!), total dipole
                        moments must be normalized by the number of
                        atoms before being passed to this function.
        charges         Training charges to fit.  Normalization must be
                        consistent with that of the dipoles.
        scalar_kernel_sparse
                        Sparse-sparse scalar kernel.  Be sure to
                        "transform" (multiply by overall scalar weight,
                        but there's a function for that) first
        scalar_kernel_transformed
                        Full-sparse transformed scalar kernel, as
                        returned by 'transform_kernels()'
        vector_kernel_sparse
                        Sparse-sparse vector kernel.  Should be
                        transformed before use, ideally using
                        'transform_sparse_kernels()'
        vector_kernel_transformed
                        Full-sparse transformed vector kernel, as
                        returned by 'transform_kernels()'
        scalar_weight   Overall scaling (delta) for scalar component
        vector_weight   Overall scaling (delta) for vector component
        charge_mode     How to treat the total charges for the scalar
                        and combined fits.  Options are: 'none' (ignore
                        them), 'fit' (the default: include them in the
                        fit with their own regularizations), and
                        'lagrange' (constrain them exactly using
                        Lagrange multipliers -- not recommended;
                        generally gives poor results with environment
                        sparsification.  Also not implemented in
                        combination with vector fits.)
        dipole_regularization
                        Regularizer for the dipole components, in dipole
                        units (normalizing the dipole effectively scales
                        this regularizer by the number of atoms)
        charge_regularization
                        Regularizer for the total charge, in charge units
        sparse_jitter   Small positive constant to add to the diagonal
                        of the sparse covariance matrix.  No longer
                        recommended; see 'condition_cutoff' instead.
        print_condition_number
                        Print the condition number of the sparse
                        covariance matrix (the matrix to be inverted)?
        condition_cutoff
                        Condition number cutoff for the stable
                        pseudoinverse linear solver ('np.linalg.lstsq');
                        this parameter is passed to that function as the
                        'rcond' parameter.  Default None.
    """
    if (scalar_weight == 0) and (vector_weight == 0):
        raise ValueError("Both weights set to zero, can't fit with no data.")
    elif ((charge_mode != 'fit') and (scalar_weight != 0)
                                 and (vector_weight != 0)):
        raise ValueError("Combined fitting only works with 'fit' charge-mode")
    if charge_mode == 'fit' and scalar_weight == 0:
        charge_mode = 'none'
        logger.warning("Requested vector kernel fitting with charges; since "
                       "l=1 kernels are insensitive to scalars, this is "
                       "exactly the same as vector fitting without charges.")
    if charge_mode == 'none':
        regularizer = dipole_regularization**-2 * np.ones((dipoles.size,))
    else:
        regularizer = solvers.make_inv_reg_vector(dipole_regularization,
                                                  charge_regularization,
                                                  len(charges))
    if charge_mode == 'none':
        if vector_weight == 0:
            scalar_kernel_transformed = np.delete(
                    scalar_kernel_transformed, slice(None, None, 4), axis=0)
            return solvers.compute_weights(
                    dipoles, scalar_kernel_sparse,
                    scalar_kernel_transformed, regularizer,
                    sparse_jitter,
                    print_condition=print_condition_number,
                    condition_cutoff=condition_cutoff)
        elif scalar_weight == 0:
            vector_kernel_transformed = np.delete(
                    vector_kernel_transformed, slice(None, None, 4), axis=0)
            return solvers.compute_weights(
                    dipoles, vector_kernel_sparse,
                    vector_kernel_transformed, regularizer,
                    sparse_jitter,
                    print_condition=print_condition_number,
                    condition_cutoff=condition_cutoff)
        else:
            raise ValueError("Can't do combined fitting without charges")
    elif charge_mode == 'fit':
        if vector_weight == 0:
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
                        vector_kernel_sparse, vector_kernel_transformed,
                        regularizer, sparse_jitter,
                        print_condition=print_condition_number,
                        condition_cutoff=condition_cutoff)
    elif charge_mode == 'lagrange':
        if vector_weight != 0:
            raise ValueError("Charge constraints not yet implemented together "
                             "with vector fitting")
        weights = solvers.compute_weights_charge_constrained(
                charges, dipoles,
                scalar_kernel_sparse, scalar_kernel_transformed, regularizer,
                sparse_jitter, condition_cutoff=condition_cutoff)
        return weights
    else:
        raise ValueError("Unrecognized charge mode '{:s}'".format(charge_mode))


def compute_residuals(
        weights, dipoles, charges, natoms_list,
        scalar_kernel_transformed, vector_kernel_transformed,
        scalar_weight=0, vector_weight=0, charge_mode='none',
        intrinsic_variation=None, print_residuals=True, write_residuals=None,
        dipole_normalized=False, **extra_args):
    """Compute the residuals of a fitted model

    The per-molecule errors are returned, as are two summary statistics:
    The per-atom RMSE is the RMSE of the norm of the dipole moment
    errors, normalized by the number of atoms in each molecule.  The
    norm MAE is the MAE of the error of the dipole prediction norms, and
    is not normalized by the number of atoms.  The charge RMSE, if
    applicable, is analogous to the per-atom dipole RMSE.

    Whether the per-molecule errors themselves are normalized by the
    number of atoms depends on whether the predicted and test dipoles
    are normalized, as indicated by the 'dipole_normalized' parameter.
    Note that the normalization must be consistent between prediction
    (model) and test!  In the case that the errors are normalized, the
    'dipole_scaling' field is set to the list of the number of atoms for
    each test geometry; otherwise, it is set to all ones.

    Parameters
        weights         Weights calculated from the fit
        dipoles         Test dipoles against which to compute residuals
        charges         Test charges against which to compute residuals
        natoms_list     List containing the number of atoms for each
                        geometry in the test set
        scalar_kernel_transformed
                        Transformed scalar full-sparse (TM) kernel of
                        the _test_ set against the sparse points
        scalar_weight   Overall scaling for the scalar component; must
                        match what was used in the fit
        vector_kernel_transformed
                        Transformed vector full-sparse (TM) kernel of
                        the _test_ set against the sparse points
        vector_weight   Overall scaling for the vector component; must
                        match what was used in the fit
        charge_mode     Charge fitting mode that was used in the fit;
                        see 'compute_weights()' for details (default
                        'none')
        intrinsic_variation
                        Intrinsic variation of the dipole moments to
                        use, instead of the RMS of the norm of the
                        test dipole moments
        print_residuals Whether to print the residual RMSEs to stdout
        write_residuals If not None, write the residuals dictionary to
                        the given filename in NumPy zipped (.npz) format
        dipole_normalized
                        Whether the given dipoles (and model!) are
                        normalized by the number of atoms

    Return a dictionary containing the following keys:
        dipole_residuals    Per-atom residuals, normalized as described
        charge_residuals    above
        dipole_scaling      Scaling to turn the dipole and charge
                            residuals into per-molecule values (see
                            above note on normalization)
        dipole_rmse         Dipole RMSE, per-atom
        dipole_frac         Dipole RMSE divided by the intrinsic
                            variation
        charge_rmse         Same as above for charges (intrinsic
        charge_frac         variation automatically calculated)
        dipole_norms_predicted
        dipole_norms_test   Norms of the predicted and test dipoles, per
                            molecule
        dipole_norm_mae     Dipole norm MAE, per molecule
    """
    if charge_mode != 'none' and scalar_weight != 0:
        charges_test = charges
    else:
        charges_test = None
        scalar_kernel_transformed = np.delete(
                scalar_kernel_transformed, slice(None, None, 4), axis=0)
        vector_kernel_transformed = np.delete(
                vector_kernel_transformed, slice(None, None, 4), axis=0)
    if vector_weight == 0:
        kernels = scalar_kernel_transformed
    elif scalar_weight == 0:
        kernels = vector_kernel_transformed
    else:
        kernels = np.concatenate((scalar_kernel_transformed,
                                  vector_kernel_transformed), axis=1)
    residuals = solvers.compute_residuals(
        weights, kernels, dipoles, natoms_list,
        charges_test=charges_test, return_rmse=True, return_norm_mae=True,
        intrinsic_dipole_std=intrinsic_variation,
        dipole_normalized=dipole_normalized)
    if print_residuals:
        # TODO violates DRY; make a function
        if 'dipole_rmse' in residuals:
            first_line = True
            for rmse, frac in np.nditer((residuals['dipole_rmse'],
                                         residuals['dipole_frac'])):
                if first_line:
                    prefix = "Dipole RMSE(s): "
                else:
                    prefix = "                "
                print("{:s}{:.10f} : {:.10f} of intrinsic variation".format(
                        prefix, rmse, frac))
                first_line = False
        if 'charge_rmse' in residuals:
            first_line = True
            for rmse, frac in np.nditer((residuals['charge_rmse'],
                                         residuals['charge_frac'])):
                if first_line:
                    prefix = "Charge RMSE(s): "
                else:
                    prefix = "                "
                print("{:s}{:.10f} : {:.10f} of intrinsic variation".format(
                        prefix, rmse, frac))
                first_line = False
    if write_residuals is not None:
        np.savez(write_residuals, **residuals)
    return residuals


def compute_per_atom_properties(
        geometries, weights,
        scalar_kernel, vector_kernel,
        scalar_weight=0, vector_weight=0, transpose_kernels=False,
        spherical=False, write_properties=None, write_properties_geoms=None):
    """Compute a per-atom breakdown of the model's predictions

    Parameters:
        geometries      List of ASE-compatible Atoms objects on which to
                        make predictions
        weights         Weights calculated from the fit
        scalar_kernel   Scalar kernel (full-sparse, _not_ transformed)
                        of the test geometries with the sparse points
        vector_kernel   Vector kernel (full-sparse, _not_ transformed,
                        atomic) of the test geometries with the sparse
                        points
        scalar_weight   Overall scaling for the scalar component; must
                        match what was used in the fit
        vector_weight   Overall scaling for the vector component; must
                        match what was used in the fit
        transpose_kernels
                        Whether the full-sparse kernels were stored in
                        transposed (sparse-full) order
        spherical       Whether the vector kernels are in spherical
                        tensor ordering, rather than Cartesian
        write_properties
                        If not None, write the properties in NumPy
                        zipped format (.npz) to the given filename.  The
                        file will contain two arrays, 'charges' and
                        'dipoles', each with the first dimension
                        corresponding to the flattened list of all atoms
                        in the test set
        write_properties_geoms
                        If not None, write the properties together with
                        the geometries, e.g. in Extended-XYZ format, to
                        the given filename.

    """
    if scalar_weight != 0:
        scalar_kernel_transformed = transform_envts_partial_charges(
                geometries, scalar_kernel, transpose_kernels) * scalar_weight
        n_sparse = scalar_kernel_transformed.shape[1]
    if vector_weight != 0:
        vector_kernel_transformed = transform_vector_envts_atomic_dipoles(
                geometries, vector_kernel,
                transpose_kernels, spherical) * vector_weight
    per_atom_properties = dict()
    if vector_weight == 0:
        per_atom_properties['charges'] = scalar_kernel_transformed.dot(weights)
    elif scalar_weight == 0:
        per_atom_properties['dipoles'] = vector_kernel_transformed.dot(
                weights).reshape((-1, 3))
    else:
        per_atom_properties['charges'] = scalar_kernel_transformed.dot(
                weights[:n_sparse])
        per_atom_properties['dipoles'] = vector_kernel_transformed.dot(
                weights[n_sparse:]).reshape((-1, 3))
    if write_properties is not None:
        np.savez(write_properties, **per_atom_properties)
    if write_properties_geoms is not None:
        atom_index = 0
        for geom in geometries:
            n_atoms = len(geom)
            for key, prop in per_atom_properties.items():
                geom.arrays[key] = prop[atom_index : atom_index + n_atoms]
            atom_index += n_atoms
        ase.io.write(write_properties_geoms, geometries)

