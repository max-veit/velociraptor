"""Some utilities useful for the fitting step of dipole learning"""

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


def compute_weights(charges_train, dipoles_train, molecules_train,
                    descriptor_matrix, reg_matrix_inv_diag,
                    sparse_envt_idces=None, sparse_jitter=1E-8,
                    do_rank_check=True):
    """Compute the weights to fit the given data

    Parameters:
        charges_train       Training data: Charges (one per molecule)
        dipoles_train       Training data: Dipoles (")
        molecules_train     List of ASE Atoms objects containing the atomic
                            coordinates of the molecules in the training set
        descriptor_matrix   Matrix of descriptors (one row per environment)
        reg_matrix_inv_diag Diagonal of the inverse regularization matrix
        sparse_envt_idces   Indices of sparse environments to choose
        sparse_jitter       Constant diagonal to add to the sparse covariance
                            matrix to make up for rank deficiency
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
    cov_matrix_transformed = transform_envts_charge_dipoles(
            molecules_train, sparse_descriptor_matrix).dot(
                    sparse_descriptor_matrix.T)
    weights = np.linalg.solve(
        sparse_cov_matrix
        + (cov_matrix_transformed.T * reg_matrix_inv_diag).dot(
            cov_matrix_transformed),
        cov_matrix_transformed.T.dot(
            charges_dipoles_train * reg_matrix_inv_diag))
    return weights


