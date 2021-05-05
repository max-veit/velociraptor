"""Utilities for building kernels using librascal

In contrast to the older SOAPFAST interface, these functions are designed to
compute and use power spectra in memory without incurring the overhead of
reading from and writing to disk.

Public functions (to be implemented):
    compute_power_spectra  Compute power spectra needed to build kernels
                           (also does sparsification)
    compute_scalar_kernel    Compute scalar kernel from power spectra
    compute_vector_kernel    Compute vector kernel from power spectra
    recompute_scalar_kernels Compute scalar kernels from scratch
    recompute_vector_kernels Compute vector kernels from scratch
    get_predictions          Get predictions for a new structure, from scratch
May be implemented in the future:
    compute_residual         Compute the (CV-)residual given kernel params

Copyright © 2021 Max Veit.
This code is licensed under the GPL, Version 3; see the LICENSE file for
more details.
"""


import logging
from math import sqrt
import os
import subprocess

import ase.io
import numpy as np

from rascal.representations import SphericalInvariants, SphericalCovariants
from rascal.models import Kernel

#  from ..fitutils import (transform_kernels, transform_sparse_kernels,
                        #  compute_weights, compute_residuals, get_charges)

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
        geometries, n_max, l_max, atom_width, r_cut, r_cut_width, rad_r0, rad_m,
        lambda_=0, n_sparse_envs=None, n_sparse_components=None, feat_sparsefile=None):
    """Compute invariant or covariant power spectra using librascal

    The arguments correspond to hyperparameters of the SOAP descriptor
    documented in rascal.representations.Spherical[In,Co]variants, and
    are summarized below. (TODO)

    Parameters
    ----------

        geometries: list(ase.Atoms)
            Atomic configurations to compute the descriptors for

    Return a librascal AtomsList that can provide the power spectra as one
    large array, or one array per species-pair block.
    """
    #TODO spline optimization?
    hypers = {
            "max_radial": n_max,
            "max_angular": l_max,
            "gaussian_sigma_constant": atom_width,
            "gaussian_sigma_type": "Constant",
            "interaction_cutoff": r_cut,
            "cutoff_smooth_width": r_cut_width,
            "cutoff_function_type": "RadialScaling",
            "cutoff_function_parameters": {
                "rate": 1.0,
                "scale": rad_r0,
                "exponent": rad_m
            }
    }
    if lambda_ == 0:
        Rep = SphericalInvariants
    else:
        Rep = SphericalCovariants
        hypers["covariant_lambda"] = lambda_
    calculator = Rep(**hypers)
    return calculator.transform(geometries)


def compute_scalar_kernel(ps, ps_other=None, zeta=2):
    """Compute the scalar (lambda=0) kernel using two power spectra"""
    pass


def compute_vector_kernel(ps, ps_other=None, ps0=None, ps0_other=None):
    """Compute the vector (lambda=1) kernel from power spectra

    If zeta is greater than 1, then must also provide scalar power
    spectra to compute covariant polynomial kernel
    """
    if (zeta != 1.0) and (ps0 is None):
        raise ValueError("Must provide a scalar power spectrum for zeta != 1")
    raise RuntimeError("Not yet implemented")


def get_predictions(geom, hypers, sparsepoints, weights):
    """Get predictions for a new structure with the given model"""
    pass
