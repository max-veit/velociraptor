#!/usr/bin/env python
"""Script to compute the residuals of a previously-stored model"""


import argparse
import logging
import os

import ase.io
import numpy as np

from velociraptor.fitutils import (transform_kernels, compute_residuals,
                                   get_charges)


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Get the residuals for a previously-stored model on new data",
    epilog="Charges are assumed to sum to zero for each geometry, unless "
    "the geometries have a property (info entry, in ASE terminology) named "
    "'total_charge'.")
#TODO(max) reduce some of this duplication with parent parsers
parser.add_argument(
    'geometries', help="Geometries of the molecules in the test set; must be "
            "the name of a file readable by ASE.")
parser.add_argument(
    'dipoles', help="Dipoles of the test set, in Cartesian coordinates, per "
            "geometry.  Entries must be in the same order as the "
            "geometry file.")
parser.add_argument(
    'weights', help="Filename where the model weights are stored")
parser.add_argument(
    'scalar_kernel', help="Filename for the full-sparse (NM) scalar kernel, "
            "in atomic environment space")
parser.add_argument(
    'tensor_kernel', help="Filename for the full-sparse tensor kernel, "
            "mapping Cartesian components to environments")
parser.add_argument(
    '-ws', '--scalar-weight', type=float, metavar='weight',
            help="Weight of the scalar component (charges) in the model",
            required=True)
parser.add_argument(
    '-wt', '--tensor-weight', type=float, metavar='weight',
            help="Weight of the tensor component (point dipoles) in the model",
            required=True)
parser.add_argument(
    '-dn', '--dipole-not-normalized',
            action='store_false', dest='dipole_normalized',
            help="Were the dipoles NOT normalized by the number of atoms "
            "before fitting? (make sure this option is consistent between fit "
            "and prediction!)")
parser.add_argument(
    '-pr', '--print-residuals', action='store_true', help="Print the RMSE "
            "residuals of the model evaluated on the test set")
parser.add_argument(
    '-wr', '--write-residuals', metavar='FILE', help="File in which to write "
            "the individual (non-RMSed) residuals.  "
            "If not given, these will not be written.")
parser.add_argument(
    '-iv', '--intrinsic-variation', metavar='sigma_0', type=float,
            help="Supply an intrinsic variation to use when normalizing "
            "dipole residual RMSEs, instead of the standard deviation of the "
            "norm of the dipole vector")
parser.add_argument(
    '-tk', '--transpose-full-kernels', action='store_true', help="Transpose "
            "the full-sparse kernels, assuming they were stored in the "
            "opposite order (MN) from the one expected (NM) (where N is full "
            "and M is sparse)")
parser.add_argument(
    '-tvk', '--transpose-vector-kernels', action='store_true', help="Same as "
            "-tk, but only for the (full) vector kernels.  This is a "
            "transitional option to patch up different conventions; don't "
            "expect it to stick around.")
parser.add_argument(
    '-tm', '--tensor-kernel-molecular', action='store_true', help="Is the full"
            " tensor kernel stored in molecular, rather than atomic, format? "
            "(i.e. are they pre-summed over the atoms in a molecule?) "
            "Note this option is compatible with -tk and -tvk.")
parser.add_argument(
    '-st', '--spherical-tensor-ordering', action='store_true',
           dest='spherical', help="Transform the vector kernels from spherical"
           " tensor to the internal Cartesian ordering")


def load_kernels(args):
    if args.scalar_weight != 0:
        scalar_kernel = np.load(args.scalar_kernel)
    else:
        scalar_kernel = np.array([])
    if args.tensor_weight != 0:
        tensor_kernel = np.load(args.tensor_kernel)
    else:
        tensor_kernel = np.array([])
    del args.scalar_kernel
    del args.tensor_kernel
    return (scalar_kernel, tensor_kernel)


if __name__ == "__main__":
    args = parser.parse_args()
    if (not args.print_residuals) and (args.write_residuals is None):
        raise ValueError("You don't want to print or write residuals, so "
                         "there's nothing for me to do.")
    scalar_kernel, tensor_kernel = load_kernels(args)
    geometries = ase.io.read(args.geometries, ':')
    natoms_list = [geom.get_number_of_atoms() for geom in geometries]
    scalar_kernel_transformed, tensor_kernel_transformed = transform_kernels(
            geometries, scalar_kernel, args.scalar_weight,
            tensor_kernel, args.tensor_weight, args.tensor_kernel_molecular,
            args.transpose_full_kernels,
            args.dipole_normalized, args.spherical)
    charges = get_charges(geometries)
    dipole_fext = os.path.splitext(args.dipoles)[1]
    if dipole_fext == '.npy':
        dipoles = np.load(args.dipoles)
    elif (dipole_fext == '.txt') or (dipole_fext == '.dat'):
        dipoles = np.loadtxt(args.dipoles)
    else:
        logger.warn("Dipoles file has no filename extension; assuming "
                    "plain text.")
        dipoles = np.loadtxt(args.dipoles)
    if args.dipole_normalized:
        dipoles = (dipoles.T / natoms_list).T
        charges = charges / natoms_list
    weights = np.load(args.weights)
    del args.dipoles
    del args.weights
    args.charge_mode = 'fit'
    compute_residuals(weights, dipoles, charges, natoms_list,
                      scalar_kernel_transformed,
                      tensor_kernel_transformed, **vars(args))

