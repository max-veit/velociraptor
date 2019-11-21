#!/usr/bin/env python
"""Compute the cross-validation error of a given model on a dataset"""


import argparse
import logging
import os

import ase.io
import numpy as np

from velociraptor.fitutils import (transform_kernels, transform_sparse_kernels,
                                   compute_weights, compute_residuals, get_charges)


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Compute the cross-validation (CV) error of the model "
    "with specified kernel parameters on a dataset",
    epilog="See 'do_fit.py' for more explanation of the fitting options.")
parser.add_argument(
    'geometries', help="Geometries of the molecules in the fit; should be the "
            "name of a file readable by ASE.")
parser.add_argument(
    'dipoles', help="Dipoles, in Cartesian coordinates, per geometry.  "
            "Entries must be in the same order as the geometry file.")
# Kernel parameters
#TODO package these into their own sub-parser?
parser.add_argument(
    '-n', '--max-radial', type=int, metavar='N', help="Number of radial "
            "channels for the SOAP kernel", default=8)
parser.add_argument(
    '-l', '--max-angular', type=int, metavar='L', help="Maximum angular "
            "momentum number for the SOAP kernel", default=6)
parser.add_argument(
    '-nsf', '--num-sparse-features', type=int, metavar='N_F', help="Sparsify "
            "to this many feature vector components", default=-1)
parser.add_argument(
    '-nse', '--num-sparse-environments', type=int, metavar='N_E',
            help="Sparsify to this many unique environments", default=-1)
parser.add_argument(
    '-aw', '--atom-sigma', type=float, metavar='sigma', help="Width of the "
            "atomic Gaussian smearing", default=0.3)
parser.add_argument(
    '-c', '--cutoff', type=float, metavar='r_c', help="Spherical cutoff of "
            "the representation", default=5.0)
parser.add_argument(
    '-r0', '--radial-scaling-scale', type=float, metavar='r_0', help="Scale "
            "parameter for the radial scaling function")
parser.add_argument(
    '-rm', '--radial-scaling-power', type=float, metavar='m', help="Scaling "
            "exponent for the radial scaling function")
parser.add_argument(
    '-ws', '--scalar-weight', type=float, metavar='weight',
            help="Weight of the scalar component (charges) in the model",
            required=True)
parser.add_argument(
    '-wt', '--tensor-weight', type=float, metavar='weight',
            help="Weight of the tensor component (point dipoles) in the model",
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
    '-k', '--cv-num-partitions', type=int, metavar='k', help="Do k-fold cross "
            "validation, i.e. split the dataset into k randomly-chosen and "
            "more-or-less equal partitions and do cross validation with them")
parser.add_argument(
    '-orc', '--optimize-charge-reg', action='store_true', help="Optimize the "
            "CV-error w.r.t. the charge regularization?")
parser.add_argument(
    '-ord', '--optimize-dipole-reg', action='store_true', help="Optimize the "
            "CV-error w.r.t. the dipole regularization?")
parser.add_argument(
    '-opt', '--optimize-all', action='store_true', help="Optimize the CV-error"
            " w.r.t all of the numerical, non-integer kernel parameters?")
parser.add_argument(
    '-nt', '--num-training-geometries', type=int, nargs='*', metavar='<n>',
            help="Keep only the first <n> geometries for training.  "
            "Specify multiple values to do a learning curve.")
parser.add_argument(
    '-pr', '--print-residuals', action='store_true', help="Print the RMSE "
            "residuals of the model evaluated on its own training data")
parser.add_argument(
    '-wr', '--write-residuals', metavar='FILE', help="File in which to write "
            "the individual (non-RMSed) residuals.  If not given, these will "
            "not be written.")
parser.add_argument(
    '-wk', '--save-kernels', metavar='PREFIX', help="Save the kernels to files"
            " with the given prefix")
parser.add_argument(
    '-wd', '--working-directory', metavar='DIR', help="Working directory for "
            "power spectrum and kernel computations (geometry and dipole files"
            " are interpreted relative to this directory if paths are not "
            "otherwise specified")
