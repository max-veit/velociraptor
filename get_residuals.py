#!/usr/bin/env python
"""Script to compute the residuals of a previously-stored model"""


import argparse
import logging

import ase
import numpy as np

import do_fit
import fitutils
import transform


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Get the residuals for a previously-stored model on new data",
    epilog="Charges are assumed to sum to zero for each geometry, unless "
    "the geometries have a property (info entry, in ASE terminology) named "
    "'total_charge'.")
#TODO(max) reduce some of this duplication with parent parsers
parser.add_argument('geometries', help="Geometries of the molecules in the "
    "test set; should be the name of a file readable by ASE.")
parser.add_argument('dipoles', help="Dipoles of the test set, in Cartesian "
    "coordinates, per geometry.  Entries must be in the same order as the "
    "geometry file.")
parser.add_argument('weights', help="Filename where the model weights are "
                                    "stored")
parser.add_argument('scalar_kernel', help="Filename for the full-sparse "
    "(NM) scalar kernel, in atomic environment space")
parser.add_argument('tensor_kernel', help="Filename for the "
    "full-sparse tensor kernel, mapping Cartesian components to environments")
parser.add_argument('-ws', '--scalar-weight', type=float, metavar='weight',
    help="Weight of the scalar component (charges) in the model",
    required=True)
parser.add_argument('-wt', '--tensor-weight', type=float, metavar='weight',
    help="Weight of the tensor component (point dipoles) in the model",
    required=True)
parser.add_argument('-pr', '--print-residuals', action='store_true',
                    help="Print the RMSE residuals of the model evaluated on "
                    "its own training data")
parser.add_argument('-wr', '--write-residuals', metavar='FILE',
                    help="File in which to write the individual (non-RMSed) "
                    "residuals.  If not given, these will not be written.")
parser.add_argument('-iv', '--intrinsic-variation', metavar='sigma_0',
                    type=float, help="Supply an intrinsic variation to use "
                    "when normalizing dipole residual RMSEs, instead of the "
                    "standard deviation of the norm of the dipole vector")


def load_kernels(args):
    if args.scalar_weight != 0:
        # Aargh, the ordering conventions changed
        scalar_kernel = np.load(args.scalar_kernel).T
    else:
        scalar_kernel = np.array([])
    if args.tensor_weight != 0:
        tensor_kernel = np.load(args.tensor_kernel).swapaxes(0, 1)
    else:
        tensor_kernel = np.array([])
    return (scalar_kernel, tensor_kernel)


if __name__ == "__main__":
    args = parser.parse_args()
    if (not args.print_residuals) and (args.write_residuals is None):
        raise ValueError("You don't want to print or write residuals, so "
                         "there's nothing for me to do.")
    scalar_kernel, tensor_kernel = load_kernels(args)
    geometries = ase.io.read(args.geometries, ':')
    natoms_list = [geom.get_number_of_atoms() for geom in geometries]
    (scalar_kernel_transformed,
     tensor_kernel_transformed) = do_fit.transform_kernels(
                                geometries, scalar_kernel, args.scalar_weight,
                                            tensor_kernel, args.tensor_weight)
    charges = do_fit.get_charges(geometries)
    dipoles = np.loadtxt(args.dipoles)
    weights = np.load(args.weights)
    args.charge_mode = 'fit'
    do_fit.compute_own_residuals(args, weights, dipoles, charges, natoms_list,
                                 scalar_kernel_transformed,
                                 tensor_kernel_transformed)

