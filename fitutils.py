"""Some utilities useful for the fitting step of dipole learning

TODO include the Lagrange multiplier method to keep charges neutral
"""

import logging
import numpy as np

import transforms


logger = logging.getLogger(__name__)


def make_reg_vector(dipole_sigma, charge_sigma, n_train):
    """Compute the diagonal of the regularization matrix.

    (much more space-efficient than computing the whole matrix, because
    numpy also stores all the zeroes -- even for diagonal matrices)
    """
    reg_matrix = np.empty((n_train, 4))
    reg_matrix_inv = np.empty((n_train, 4))
    reg_matrix[:, 0] = charge_sigma**2
    reg_matrix_inv[:, 0] = charge_sigma**-2
    reg_matrix[:, 1:] = dipole_sigma**2
    reg_matrix_inv[:, 1:] = dipole_sigma**-2
    reg_matrix = reg_matrix.reshape(-1)
    reg_matrix_inv = reg_matrix_inv.reshape(-1)
    return reg_matrix, reg_matrix_inv


def merge_charges_dipoles(charges, dipoles):
    """Merge two arrays of (total!) charges and dipoles

    (for the same set of atoms, using the canonical ordering)

    The first index should index the configuration, the second
    (for dipoles) the Cartesian axis.
    """
    n_train = len(charges)
    charges_dipoles = np.concatenate((charges, dipoles), axis=1)
    return charges_dipoles.reshape(n_train*4)


def split_charges_dipoles(charges_dipoles):
    """Split a combined charges-dipoles array into two

    Return a tuple containing the charges and (2-D) diplole array

    This is effectively the inverse of merge_charges_dipoles().
    """
    n_train = charges_dipoles.size / 4
    charges_dipoles = charges_dipoles.reshape((n_train, 4))
    return charges_dipoles[:,0], charges_dipoles[:, 1:4]


def compute_residuals(weights, kernel_matrix, dipoles_test, geoms_test,
                      charges_included=True, return_rmse=True):
    """Compute the residuals for the given fit

    If the RMSE is requested, return the RMS of the _norm_ of the dipole
    residuals, normalized (for each geometry) by the corresponding
    number of atoms.  For charges, the RMSE of the total charge residual
    is returned, normalized again by the number of atoms in the geometry.

    Parameters:
        weights     Weights computed for the fit
        kernel_matrix
                    Kernel between model basis functions (rows) and test
                    dipoles, optionally with charges.  Order should be
                    (charge), x, y, z, per configuration.
        dipoles_test
                    Computed ("exact") dipoles for the test set
        geoms_test  List of ASE-compatible Atoms objects containing the
                    atomic coordinates of all the geometries in the test
                    set
        charges_included
                    Whether (total) charges are included in the kernel
                    matrix above.  If true, the charge residual is
                    computed and returned separately from the dipoles.
                    Computed/exact total charges are extracted from
                    geoms_test ('total_charge' property), default 0.
        return_rmse Whether to return the RMS errors to summarize the
                    residuals

    Return value is either a numpy array or a tuple.  Depending on what
    is requested, the order will be one of:
        dipole_residuals
        (dipole_residuals, charge_residuals)
        (dipole_residuals, dipole_rmse)
        (dipole_residuals, charge_residuals, dipole_rmse, charge_rmse)
    """
    if charges_included:
        charges_test = [geom.info.get('total_charge', 0.)
                        for geom in geoms_test]
        data_test = merge_charges_dipoles(charges_test, dipoles_test)
    else:
        data_test = dipoles_test
    natoms_test = np.array([geom.get_number_of_atoms() for geom in geoms_test])
    n_test = len(geoms_test)
    residuals = weights.dot(kernel_matrix) - data_test
    if charges_included:
        charge_residuals, dipole_residuals = split_charges_dipoles(residuals)
    if return_rmse:
        dipole_rmse = np.sqrt(np.sum((dipole_residuals / natoms_test)**2)
                              / n_test)
        if charges_included:
            charge_rmse = np.sqrt(np.sum((charge_residuals / natoms_test)**2)
                                  / n_test)
            return (dipole_residuals, charge_residuals,
                    dipole_rmse, charge_rmse)
        else:
            return (dipole_residuals, dipole_rmse)
    else:
        if charges_included:
            return (dipole_residuals, charge_residuals)
        else:
            return dipole_residuals


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


def compute_weights(charges_train, dipoles_train, molecules_train,
                    descriptor_matrix, reg_matrix_inv_diag,
                    sparse_envt_idces=None, sparse_jitter=1E-8,
                    kernel_power=1, do_rank_check=True):
    """Compute the weights to fit the given data

    Parameters:
        charges_train       Training data: Charges (one per molecule)
        dipoles_train       Training data: Dipoles (") (flattened)
        molecules_train     List of ASE Atoms objects containing the atomic
                            coordinates of the molecules in the training set
        descriptor_matrix   Matrix of descriptors (one row per environment)
        reg_matrix_inv_diag Diagonal of the inverse regularization matrix
        sparse_envt_idces   Indices of sparse environments to choose
        sparse_jitter       Constant diagonal to add to the sparse covariance
                            matrix to make up for rank deficiency
        kernel_power        Optional element-wise exponent to sharpen the
                            kernel
        do_rank_check       Check the rank of the sparse covariance matrix
                            to make sure it's (relatively) well-conditioned?
                            (default True)

    Concretely, this computes the kernel matrix as K = D^T D (where D is the
    descriptor matrix) and then computes the weights to minimize the
    loss function:

        || L K x - y ||^2_Lambda + || x ||^2_K

    and L transforms the space of environments to charges and dipoles.

    In-memory version: Make sure the descriptor matrix is large enough
    to fit in memory!  (offline version coming soon)
    """
    charges_dipoles_train = merge_charges_dipoles(charges_train, dipoles_train)
    sparse_cov_matrix, cov_matrix_transformed = compute_cov_matrices(
            molecules_train, sparse_envt_idces, sparse_jitter,
            kernel_power, do_rank_check)
    weights = np.linalg.solve(
        sparse_cov_matrix
        + (cov_matrix_transformed.T * reg_matrix_inv_diag).dot(
            cov_matrix_transformed),
        cov_matrix_transformed.T.dot(
            charges_dipoles_train * reg_matrix_inv_diag))
    return weights


def compute_weights_charge_constrained(
                    charges_train, dipoles_train, molecules_train,
                    descriptor_matrix, reg_matrix_inv_diag,
                    sparse_envt_idces=None, sparse_jitter=1E-8,
                    kernel_power=1, do_rank_check=True):
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
        dipoles_train       Training data: Dipoles (") (flattened)
        molecules_train     List of ASE Atoms objects containing the atomic
                            coordinates of the molecules in the training set
        descriptor_matrix   Matrix of descriptors (one row per environment)
        reg_matrix_inv_diag Diagonal of the inverse regularization matrix
        sparse_envt_idces   Indices of sparse environments to choose
        sparse_jitter       Constant diagonal to add to the sparse covariance
                            matrix to make up for rank deficiency
        kernel_power        Optional element-wise exponent to sharpen the
                            kernel
        do_rank_check       Check the rank of the sparse covariance matrix
                            to make sure it's (relatively) well-conditioned?
                            (default True)

    Returns the weights as well as the values of the Lagrange multipliers
    """
    sparse_cov_matrix, cov_matrix_transformed = compute_cov_matrices(
            molecules_train, sparse_envt_idces, sparse_jitter,
            kernel_power, do_rank_check)
    cov_matrix_charges = cov_matrix_transformed[::4]
    cov_matrix_dipoles = np.delete(cov_matrix_transformed,
        np.arange(0, cov_matrix_transformed.shape[0], 4), axis=0)
    # TODO still haven't found a better way to solve the equations than
    #      to construct this huge block matrix
    kernel_block = ((cov_matrix_dipoles.T * reg_matrix_inv_diag).dot(
                        cov_matrix_dipoles) + sparse_cov_matrix)
    # TODO is it possible to add a sparse jitter to the lower part of
    #      the diagonal as well? (in place of the zeros)
    n_charges = cov_matrix_charges.shape[0]
    lhs_matrix = np.block(
        [[kernel_block,       cov_matrix_charges.T],
         [cov_matrix_charges, np.zeros((n_charges, n_charges))]])
    rhs = np.concatenate(
        (cov_matrix_dipoles.T.dot(dipoles_train * reg_matrix_inv_diag),
         (charges_train)))
    results = np.linalg.solve(lhs_matrix, rhs)
    weights = results[:sparse_cov_matrix.shape[0]]
    lagrange_multipliers = results[sparse_cov_matrix.shape[0]:]
    return weights, lagrange_multipliers


def compute_weights_two_model(charges_train, dipoles_train, molecules_train,
                              reg_matrix_inv_diag,
                              scalar_kernel_sparse, scalar_kernel_transformed,
                              tensor_kernel_sparse, tensor_kernel_transformed):
    """Compute the weights for the two-model problem:

    μ_tot = x_1 K_1 + x_2 K_2

    with two different sets of weights and kernel matrices.  Here
    we specifically use the case of a sum of scalar and tensor models.

    Parameters:
        charges_train       Training data: Charges (one per molecule)
        dipoles_train       Training data: Dipoles (") (flattened)
        molecules_train     List of ASE Atoms objects containing the atomic
                            coordinates of the molecules in the training set
        reg_matrix_inv_diag Diagonal of the inverse regularization matrix
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

