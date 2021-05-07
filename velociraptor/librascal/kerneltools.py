"""Utilities for building kernels using librascal

In contrast to the older SOAPFAST interface, these functions are designed to
compute and use power spectra in memory without incurring the overhead of
reading from and writing to disk.

Public functions:
    compute_power_spectra  Compute power spectra needed to build kernels
                           (also applies pre-computed sparsification)
    compute_scalar_kernel    Compute scalar kernel from power spectra
    compute_vector_kernel    Compute vector kernel from power spectra
    get_test_kernel          Get kernel for new structures, from scratch
May be implemented in the future:
    compute_residual         Compute the (CV-)residual given kernel params

Copyright © 2021 Max Veit.
This code is licensed under the GPL, Version 3; see the LICENSE file for
more details.
"""


from collections import defaultdict
import itertools
import logging
from math import sqrt
import os
import subprocess

import ase.io
import numpy as np

from rascal.representations import SphericalInvariants, SphericalCovariants
from rascal.representations.spherical_invariants import get_power_spectrum_index_mapping
from rascal.models import Kernel

from ..fitutils import transform_kernels, compute_residuals

LOGGER = logging.getLogger(__name__)


def _get_soapfast_to_librascal_idx_mapping(n_max, l_max, n_species, lambda_=0):
    """Return a mapping from SOAPFAST to librascal descriptor indices

    The mapping is in the form of an array of the same size as the SOAPFAST
    descriptor, giving the librascal descriptor index corresponding to each
    SOAPFAST-indexed entry of the array.

    Currently only implemented for lambda=0
    (TODO: The only thing that would really change would be the species block
    size, and we know how to compute that, right?)
    """
    if lambda_ != 0:
        raise ValueError("Only implemented for lambda != 0")
    species_pair_block_size = n_max**2 * (l_max + 1)
    soapfast_to_librascal_idx_mapping = -1*np.ones((n_species, n_species, n_max, n_max, l_max + 1), dtype=int)
    species_block_base = np.arange(species_pair_block_size).reshape(n_max, n_max, l_max + 1)
    # Only explicitly iterate over the upper triangle
    block_offset = 0
    for species_a_idx in range(n_species):
        for species_b_idx in range(species_a_idx, n_species):
            soapfast_to_librascal_idx_mapping[
                species_a_idx, species_b_idx
            ] = species_block_base + block_offset
            if species_a_idx != species_b_idx:
                # Fill the corresponding (nn'-mirrored) block of the lower triangle
                soapfast_to_librascal_idx_mapping[
                    species_b_idx, species_a_idx
                ] = species_block_base.transpose((1, 0, 2)) + block_offset
            # Advance by one librascal species-block
            block_offset += species_pair_block_size
    return soapfast_to_librascal_idx_mapping.flatten()


def compute_power_spectra(
        geometries, soap_hypers,
        lambda_=0, n_sparse_envs=None, n_sparse_components=None,
        feat_sparsefile=None, Amat_file=None, species_list=None):
    """Compute invariant or covariant power spectra using librascal

    The SOAP hyperparameters are passed as arguments to the
    rascal.representations.Spherical(In|Co)variants constructors.
    The remaining parameters are summarized below.

    Parameters
    ----------

    geometries: list(ase.Atoms)
        Atomic configurations to compute the descriptors for

    lambda_: int
        Spherical tensor order for the power spectra.  Lambda=0 computes
        scalar SOAP, any other value computes spherical covariants of
        that order.

    feat_sparsefile: filename
        Name of a feature sparsification file written by SOAPFAST
        or TENSOAP (usually ending in `_fps.npy`)

    Amat_file: filename
        Name of the feature rescaling file also written out by SOAPFAST
        or TENSOAP (if not given, will try to derive from 'feat_sparsefile')

    species_list: list(int)
        List of atomic numbers for species considered by SOAPFAST
        (only used when reading feature sparsification files)
        Warning: librascal only works with sorted species lists,
        so if the SOAPFAST feature sparsification files were made
        with unsorted species lists, they should be transformed
        or remade.

    Returns
    -------
    features: np.ndarray
        Feature matrix, size NxD, where N is the number of atoms
        (environments) and D is the feature dimension If using feature
        sparsification, the features are reweighted (using the "A-matrix")
        before being returned.
    """
    if lambda_ == 0:
        Rep = SphericalInvariants
    else:
        Rep = SphericalCovariants
        hypers["covariant_lambda"] = lambda_
    if (n_sparse_components is not None) or (n_sparse_envs is not None):
        raise ValueError("Feature sparsification with librascal is not yet implemented."
                         "Please use a pre-generated sparse file instead.")
    if feat_sparsefile is not None:
        if not species_list:
            species_list = np.unique(geometries[0].get_atomic_numbers())
        species_pairs = [(species_list[i], species_list[j]) for i in range(len(species_list)) for j in range(i, len(species_list))]
        sparse_idces = np.load(feat_sparsefile)
        n_max = hypers['max_radial']
        l_max = hypers['max_angular']
        # Feature sparsification is directly implemented in SphericalInvariants
        #TODO use global_species in librascal so that the species list doesn't change?
        if lambda_=0:
            idx_mapping = _get_soapfast_to_librascal_idx_mapping(n_max, l_max, len(species_list))
            sparse_idces_rascal = idx_mapping[sparse_idces]
            rascal_index_mapping = get_power_spectrum_index_mapping(species_pairs, n_max, l_max)
            rascal_spinv_select_idces = [rascal_index_mapping[idx] for idx in sparse_idces_rascal]
            hypers['coefficient_subselection'] = rascal_spinv_select_idces
        else:
            # I would just get the features and manually pick out the columns
            raise ValueError("Feature sparsification not yet implemented for lambda != 0")
    calculator = Rep(**hypers)
    soaps = calculator.transform(geometries)
    results = soaps.get_features()
    if feat_sparsefile is not None:
        amat_file = feat_sparsefile.replace('fps', 'Amat')
        if amat_file == feat_sparsefile:
            raise ValueError("Must provide name of A-matrix file (could not "
                             "derive from feature sparsification file)")
        Amat = np.load(amat_file)
        results = results @ Amat
    return results


def apply_environment_sparsification(ps, env_sparse_file):
    """Apply a pre-computed environment sparsification selection"""
    pass


def compute_scalar_kernel(ps, ps_other=None, zeta=2):
    """Compute the scalar (lambda=0) kernel using two power spectra"""
    if ps_other is None:
        return (ps @ ps.T)**zeta
    else:
        return (ps @ ps_other.T)**zeta


def compute_vector_kernel(ps, ps_other=None, ps0=None, ps0_other=None, zeta=2):
    """Compute the vector (lambda=1) kernel from power spectra

    Vector PS shape is assumed to be (N_samp, 3, dim), where 'dim'
    is the descriptor dimension

    If zeta is greater than 1, then must also provide scalar power
    spectra to compute covariant polynomial kernel
    """
    if (zeta != 1.0) and (ps0 is None):
        raise ValueError("Must provide a scalar power spectrum for zeta != 1")
    kernel_covariant
    if ps_other is None:
        ps_other = ps
        ps0_other = ps0
    # This sums over the descriptor dimension, while keeping the sample and
    # spherical component dimensions.  The resulting matrix is rearranged to
    # "SOAPFAST order", keeping the first two sample dimensions first, then
    # the spherical component dimensions second, in the same order as the
    # sample dimensions.
    #TODO doesn't the second PS need to be "complex conjugated"?
    kernel_covariant = np.tensordot(ps, ps_other.T, axes=1).transpose((0, 3, 1, 2))
    if zeta != 1.0:
        kernel_scalar = (ps0 @ ps0_other.T)**(zeta - 1.0)
        # Covariant kernel shape is (N1, N2, 
        kernel_covariant *= kernel_scalar[:, :, np.newaxis, np.newaxis]
    return kernel_covariant


def get_test_kernel(geometries, lambda_orders, hypers, kernel_zeta,
                    PS_sparse,
                    feat_sparsefile=None, Amat_file=None):
    """Get kernel for new structures with the given model

    Parameters
    ----------

    geometries: list(ase.Atoms)
        Atomic configurations to get kernel for

    lambda_orders: list(int)
        Spherical tensor orders needed for the power spectra.  For a pure scalar
        kernel this is just 0, for a vector kernel generally it's [0, 1].

    hypers: list(dict)
        Dictionary of SOAP hyperparameters, one for each entry of lambda_orders

    kernel_zeta: int
        Power to raise the SOAP kernel to

    PS_sparse: list(2-D array)
        Power spectra for the model's "sparse" or "basis" points
        (shape N_sparse (x N_comp) x N_descriptor), one PS for each entry
        of lambda_orders)
        The first dimension of each PS must be equal

    feat_sparsefile: list(filename), optional
        Name of a feature sparsification file written by SOAPFAST
        or TENSOAP (usually ending in `_fps.npy`)
        One for each entry of lambda_orders

    Amat_file: list(filename), optional
        Name of the feature rescaling file also written out by SOAPFAST
        or TENSOAP (if not given, will try to derive from 'feat_sparsefile')
        One for each entry of lambda_orders
    """
    if feat_sparsefile is None:
        feat_sparsefile = itertools.repeat(None)
    if Amat_file is None:
        Amat_file = itertools.repeat(None)
    PS_test = []
    for lambda_, lhypers, lf_sparsefile, lAmat_file in zip(
            lambda_orders, hypers, feat_sparsefile, Amat_file):
        PS_test.append(compute_power_spectra(
                geometries, lhypers, lambda_, lf_sparsefile, lAmat_file))
    if 1 in lambda_orders:
        # Vector kernel
        # Ensure we use power spectra corresponding to the correct lambda order
        # Also allows omitting lambda=0 PS if zeta==1
        PS_sparse_lambda = defaultdict(lambda: None)
        PS_test_lambda = defaultdict(lambda: None)
        for lambda_, lPS_sparse, lPS_test in zip(lambda_orders, PS_sparse, PS_test):
            PS_sparse_lambda[lambda_] = lPS_sparse
            PS_test_lambda[lambda_] = lPS_test
        return compute_vector_kernel(PS_sparse_lambda[1], PS_test_lambda[1],
                                     PS_sparse_lambda[0], PS_test_lambda[0],
                                     kernel_zeta)
    else:
        if (lambda_orders[0] != 0) or (len(lambda_orders) > 1):
            raise ValueError("Use only lambda_orders=[0,] to compute scalar kernel")
        return compute_scalar_kernel(PS_sparse[0], PS_test[0], kernel_zeta)
