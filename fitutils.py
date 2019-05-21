"""Some utilities useful for the fitting step of dipole learning"""

import logging
import numpy as np

import transform


logger = logging.getLogger(__name__)


def make_reg_vector(dipole_sigma, charge_sigma, n_train):
    """Compute the diagonal of the regularization matrix.

    (much more space-efficient than computing the whole matrix, because
    numpy also stores all the zeroes -- even for diagonal matrices)
    """
    reg_matrix = np.empty((n_train, 4))
    reg_matrix[:, 0] = charge_sigma**2
    reg_matrix[:, 1:] = dipole_sigma**2
    reg_matrix = reg_matrix.reshape(-1)
    return reg_matrix


def make_inv_reg_vector(dipole_sigma, charge_sigma, n_train):
    """Compute the diagonal of the inverse regularization matrix.

    (much more space-efficient than computing the whole matrix, because
    numpy also stores all the zeroes -- even for diagonal matrices)
    """
    reg_matrix_inv = np.empty((n_train, 4))
    reg_matrix_inv[:, 0] = charge_sigma**-2
    reg_matrix_inv[:, 1:] = dipole_sigma**-2
    reg_matrix_inv = reg_matrix_inv.reshape(-1)
    return reg_matrix_inv


def merge_charges_dipoles(charges, dipoles):
    """Merge two arrays of (total!) charges and dipoles

    (for the same set of atoms, using the canonical ordering)

    The first index should index the configuration, the second
    (for dipoles) the Cartesian axis.
    """
    n_train = len(charges)
    charges_dipoles = np.concatenate((charges[:, np.newaxis], dipoles), axis=1)
    return charges_dipoles.reshape(n_train*4)


def split_charges_dipoles(charges_dipoles):
    """Split a combined charges-dipoles array into two

    Return a tuple containing the charges and (2-D) diplole array

    This is effectively the inverse of merge_charges_dipoles().
    """
    n_train = int(charges_dipoles.size / 4)
    charges_dipoles = charges_dipoles.reshape((n_train, 4))
    return charges_dipoles[:,0], charges_dipoles[:, 1:4]


def compute_residuals(weights, kernel_matrix, dipoles_test, natoms_test,
                      charges_test=None, return_rmse=True,
                      intrinsic_dipole_std=None):
    """Compute the residuals for the given fit

    If the RMSE is requested, return the RMS of the _norm_ of the dipole
    residuals, normalized (for each geometry) by the corresponding
    number of atoms.  For charges, the RMSE of the total charge residual
    is returned, normalized again by the number of atoms in the geometry.

    Parameters:
        weights     Weights computed for the fit
                    In the case of a multi-model fit with independent
                    kernels (i.e. a block-diagonal total kernel matrix),
                    this may also be a list of arrays of weights, one
                    for each model.  The residual is computed as the
                    difference from the sum of each independent model.
        kernel_matrix
                    Kernel between model basis functions (rows) and test
                    dipoles, optionally with charges.  Order should be
                    (charge), x, y, z, per configuration.
                    As above, this may also be a list of kernel matrices
                    of the same length as 'weights'.
        dipoles_test
                    Computed ("exact") dipoles for the test set
        natoms_test List containing the number of atoms for each geometry
                    in the test set (for normalization)
    Optional arguments:
        charges_test
                    If provided, the charge residual is additionally
                    computed and returned separately from the dipoles.
        return_rmse Whether to return the RMS errors to summarize the
                    residuals, including as fractions of the intrinsic
                    errors (standard deviations per atom) of the dipoles
                    and charges.
        intrinsic_dipole_std
                    Intrinsic variation of the dipole moments to use,
                    instead of the RMS of the norm of the dipole moments
                    in 'dipole_test'

    Return value is a dictionary of numpy arrays.  Depending on what
    is requested, one or more of the following keys will be present:
        dipole_residuals, charge_residuals, dipole_rmse, charge_rmse,
        dipole_frac, charge_frac
    """
    charges_included = (charges_test is not None)
    if hasattr(weights, 'shape') and len(weights.shape) == 1:
        # Assume we've been given arrays, not lists of arrays
        if hasattr(kernel_matrix, 'shape') and len(kernel_matrix.shape) != 2:
            raise ValueError("Confused about whether you're trying to specify "
                             "one set or multiple sets of weights and kernel")
        weights = [weights]
        kernel_matrix = [kernel_matrix]
    elif len(weights) != len(kernel_matrix):
        raise ValueError("Must supply the same number of sets of weights "
                         "and kernel matrices")
    if charges_included:
        data_test = merge_charges_dipoles(charges_test, dipoles_test)
    else:
        data_test = dipoles_test.flatten()
    n_test = len(natoms_test)
    natoms_test = np.array(natoms_test)
    predicted = sum(
        (weights_one.dot(kernel_one.T)
         for weights_one, kernel_one in zip(weights, kernel_matrix)),
        np.zeros_like(data_test))
    residuals = predicted - data_test
    residuals_out = dict()
    if charges_included:
        charge_residuals, dipole_residuals = split_charges_dipoles(residuals)
        residuals_out['charge_residuals'] = charge_residuals
    else:
        dipole_residuals = residuals.reshape(n_test, 3)
    residuals_out['dipole_residuals'] = dipole_residuals
    if return_rmse:
        dipole_rmse = np.sqrt(np.sum((dipole_residuals.T / natoms_test)**2)
                              / n_test)
        residuals_out['dipole_rmse'] = dipole_rmse
        if intrinsic_dipole_std is None:
            #TODO(max) -- should this be per-atom?
            intrinsic_dipole_std = np.sqrt(
                    np.sum((dipoles_test.T / natoms_test)**2) / n_test)
        residuals_out['dipole_frac'] = dipole_rmse / intrinsic_dipole_std
        if charges_included:
            charge_rmse = np.sqrt(np.sum((charge_residuals / natoms_test)**2)
                                  / n_test)
            residuals_out['charge_rmse'] = charge_rmse
            #TODO(max) same question
            charge_std = np.std(charges_test / natoms_test)
            if charge_std == 0.:
                residuals_out['charge_frac'] = np.nan
            else:
                residuals_out['charge_frac'] = charge_rmse / charge_std
    return residuals_out


def compute_cov_matrices(molecules_train, descriptor_matrix,
                         sparse_envt_idces=None, sparse_jitter=1E-8,
                         kernel_power=1, do_rank_check=True):
    """Compute covariance (kernel) matrices for fitting

    Parameters:
        molecules_train     List of ASE Atoms objects containing the atomic
                            coordinates of the molecules in the training set
        descriptor_matrix   Matrix of descriptors (one row per environment)
        sparse_envt_idces   Indices of sparse environments to choose
        sparse_jitter       Constant diagonal to add to the sparse covariance
                            matrix to make up for rank deficiency
        kernel_power        Optional element-wise exponent to sharpen the
                            kernel
        do_rank_check       Check the rank of the sparse covariance matrix
                            to make sure it's (relatively) well-conditioned?
                            (default True)

    Returns a tuple of the sparse covariance matrix (covariance of all the
    sparse environments with each other) and the transformed covariance
    matrix (covariance between charges+dipoles and sparse environments)

    """
    if sparse_envt_idces is not None:
        sparse_descriptor_matrix = descriptor_matrix[sparse_envt_idces]
        sparse_cov_matrix = sparse_descriptor_matrix.dot(
                                sparse_descriptor_matrix.T)
        if do_rank_check:
            sparse_rank = np.linalg.matrix_rank(sparse_cov_matrix)
            if sparse_rank < sparse_cov_matrix.shape[0]:
                logger.warning("Sparse covariance matrix possibly " +
                               "rank-deficient")
    else:
        #TODO haven't really thought much about the non-sparse case.
        #     This should work, but I'm not sure it's the best way.
        sparse_descriptor_matrix = descriptor_matrix
        sparse_cov_matrix = descriptor_matrix.dot(descriptor_matrix.T)
    if kernel_power == 1:
        cov_matrix_transformed = transform_envts_charge_dipoles(
                molecules_train, descriptor_matrix).dot(
                        sparse_descriptor_matrix.T)
    else:
        sparse_cov_matrix = sparse_cov_matrix ** kernel_power
        cov_matrix_transformed = transform_envts_charge_dipoles(
                molecules_train, (descriptor_matrix.dot(
                    sparse_descriptor_matrix.T))**kernel_power)
    return sparse_cov_matrix, cov_matrix_transformed


def compute_weights(dipoles_train, kernel_sparse, kernel_transformed,
                    reg_matrix_inv_diag, sparse_jitter=1E-9):
    """Compute the weights to fit the given data (dipoles only)

    Parameters:
        dipoles_train       Training data: Dipoles (one row per molecule)
        kernel_sparse       Covariance between the sparse environments
                            of the scalar model
        kernel_transformed  Covariance between the molecular dipoles and
                            the sparse environments
        reg_matrix_inv_diag Diagonal of the inverse regularization matrix
        sparse_jitter       Constant diagonal to add to the sparse
                            covariance matrix to make up for rank
                            deficiency

    Concretely, this computes the weights to minimize the loss function:

        || L K x - y ||^2_Lambda + || x ||^2_K

    where K is the covariance between all environments and sparse
    environments, and L transforms the space of environments to dipoles.

    In-memory version: Make sure the descriptor matrix is large enough
    to fit in memory!  (offline version coming soon)
    """
    kernel_sparse[np.diag_indices_from(kernel_sparse)] += sparse_jitter
    weights = np.linalg.solve(
        kernel_sparse
        + (kernel_transformed.T * reg_matrix_inv_diag).dot(kernel_transformed),
        kernel_transformed.T.dot(dipoles_train.flat * reg_matrix_inv_diag))
    return weights


def compute_weights_charges(charges_train, dipoles_train,
                            scalar_kernel_sparse, scalar_kernel_transformed,
                            reg_matrix_inv_diag, sparse_jitter=1E-9):
    """Compute the weights to fit the given data (dipoles and charges)

    Parameters:
        charges_train       Training data: Charges (one per molecule)
        dipoles_train       Training data: Dipoles (one row per molecule)
        scalar_kernel_sparse
                            Covariance between the sparse environments
                            of the scalar model
        scalar_kernel_transformed
                            Covariance between the molecular dipoles and
                            the sparse environments of the scalar model
        sparse_jitter       Constant diagonal to add to the sparse
                            covariance matrix to make up for rank
                            deficiency

    Concretely, this computes the weights to minimize the loss function:

        || L K x - y ||^2_Lambda + || x ||^2_K

    where K is the covariance between all environments and sparse
    environments, and L transforms the space of environments to charges
    and dipoles.

    In-memory version: Make sure the descriptor matrix is large enough
    to fit in memory!  (offline version coming soon)
    """
    charges_dipoles_train = merge_charges_dipoles(charges_train, dipoles_train)
    weights = np.linalg.solve(
        scalar_kernel_sparse
        + (scalar_kernel_transformed.T * reg_matrix_inv_diag).dot(
            scalar_kernel_transformed),
        scalar_kernel_transformed.T.dot(
            charges_dipoles_train * reg_matrix_inv_diag))
    return weights


def compute_weights_charge_constrained(
                    charges_train, dipoles_train,
                    scalar_kernel_sparse, scalar_kernel_transformed,
                    reg_matrix_inv_diag, sparse_jitter=1E-9):
    """Compute the weights to find a constrained fit the given data

    Uses Lagrange multipliers to fit the total charges exactly.
    The resulting fitting equations are:

    ╭ (K_s + K^T L^T Λ^-1 L K)x + K^T G^T 𝜆 = K^T L^T Λ^-1 μ
    ┤
    ╰ G K x = q

    (where G just sums over the atoms in a molecule to obtain total
    charge)

    Parameters:
        charges_train       Training data: Charges (one per molecule)
        dipoles_train       Training data: Dipoles (one row per molecule)
        molecules_train     List of ASE Atoms objects containing the atomic
                            coordinates of the molecules in the training set
        scalar_kernel_sparse
                            Covariance between the sparse environments
                            of the scalar model
        scalar_kernel_transformed
                            Covariance between the molecular dipoles and
                            the sparse environments of the scalar model
        reg_matrix_inv_diag Diagonal of the inverse regularization matrix
        sparse_jitter       Constant diagonal to add to the sparse
                            covariance matrix to make up for rank
                            deficiency

    Returns the weights as well as the values of the Lagrange multipliers
    """
    cov_matrix_charges = scalar_kernel_transformed[::4]
    cov_matrix_dipoles = np.delete(scalar_kernel_transformed,
        np.arange(0, scalar_kernel_transformed.shape[0], 4), axis=0)
    if scalar_kernel_sparse.shape[0] < cov_matrix_charges.shape[0]:
        logger.critical("More constraints than weights; the result will not "
                        "respect the dipoles at all.")
    # TODO still haven't found a better way to solve the equations than
    #      to construct this huge block matrix
    kernel_block = ((cov_matrix_dipoles.T * reg_matrix_inv_diag).dot(
                        cov_matrix_dipoles) + scalar_kernel_sparse)
    # TODO is it possible to add a sparse jitter to the lower part of
    #      the diagonal as well? (in place of the zeros)
    n_charges = cov_matrix_charges.shape[0]
    lhs_matrix = np.block(
        [[kernel_block,       cov_matrix_charges.T],
         [cov_matrix_charges, np.zeros((n_charges, n_charges))]])
    rhs = np.concatenate(
        (cov_matrix_dipoles.T.dot(dipoles_train.flat * reg_matrix_inv_diag),
         (charges_train)))
    results = np.linalg.solve(lhs_matrix, rhs)
    weights = results[:sparse_cov_matrix.shape[0]]
    lagrange_multipliers = results[sparse_cov_matrix.shape[0]:]
    return weights, lagrange_multipliers


def compute_weights_two_model(charges_train, dipoles_train,
                              scalar_kernel_sparse, scalar_kernel_transformed,
                              tensor_kernel_sparse, tensor_kernel_transformed,
                              reg_matrix_inv_diag, sparse_jitter=1E-9):
    """Compute the weights for the two-model problem:

    μ_tot = x_1 K_1 + x_2 K_2

    with two different sets of weights and kernel matrices.  Here
    we specifically use the case of a sum of scalar and tensor models.

    Parameters:
        charges_train       Training data: Charges (one per molecule)
        dipoles_train       Training data: Dipoles (one row per molecule)
        molecules_train     List of ASE Atoms objects containing the atomic
                            coordinates of the molecules in the training set
        reg_matrix_inv_diag Diagonal of the inverse regularization matrix
        sparse_jitter       Constant diagonal to add to the sparse
                            covariance matrix to make up for rank
                            deficiency
        Kernel matrices:
        scalar_kernel_sparse
                            Covariance between the sparse environments
                            of the scalar model
        scalar_kernel_transformed
                            Covariance between the molecular dipoles and
                            the sparse environments of the scalar model
        tensor_kernel_sparse
                            Covariance between the sparse environments
                            of the tensor model
        tensor_kernel_transformed
                            Covariance between the molecular dipoles and
                            the sparse environments of the tensor model

    Returns the scalar and tensor weights combined into a single vector.
    The first n_sparse components are the scalar weights (n_sparse is the
    number of rows of the scalar kernel matrix), the rest are the tensor
    weights.
    """
    charges_dipoles_train = merge_charges_dipoles(charges_train, dipoles_train)
    scalar_block = (scalar_kernel_transformed.T.dot(
                scalar_kernel_transformed * reg_matrix_inv_diag)
            + scalar_kernel_sparse)
    tensor_block = (tensor_kernel_transformed.T.dot(
                tensor_kernel_transformed * reg_matrix_inv_diag)
            + tensor_kernel_sparse)
    off_diag_block = scalar_kernel_transformed.T.dot(
            tensor_kernel_transformed * reg_matrix_inv_diag)
    lhs_matrix = np.block([[scalar_block,     off_diag_block],
                           [off_diag_block.T, tensor_block]])
    rhs = np.concatenate((scalar_kernel_transformed,
                          tensor_kernel_transformed), axis=1).dot(
                            charges_dipoles_train * reg_matrix_inv_diag)
    weights_combined = np.linalg.solve(lhs_matrix, rhs)
    return weights_combined

