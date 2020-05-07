#!/usr/bin/env python
"""Script to compute the per-atom breakdown of a model's predictions

Copyright © 2020 Max Veit.
This code is licensed under the GPL, Version 3; see the LICENSE file for
more details.
"""


import argparse
import logging
import os

import ase.io
import numpy as np

from velociraptor.fitutils import compute_per_atom_properties


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Get the per-atom predictions for a previously-stored model.")
#TODO(max) reduce some of this duplication with parent parsers
parser.add_argument(
    'geometries', help="Geometries of the molecules in the test set; must be "
            "the name of a file readable by ASE.")
parser.add_argument(
    'weights', help="Filename where the model weights are stored")
parser.add_argument(
    'scalar_kernel', help="Filename for the full-sparse (NM) scalar kernel, "
            "in atomic environment space")
parser.add_argument(
    'vector_kernel', help="Filename for the full-sparse vector kernel, mapping"
            " Cartesian components to environments.  Must be in atomic format,"
            " i.e. one row per atom (where each entry is a 3x3 matrix).")
parser.add_argument(
    '-ws', '--scalar-weight', type=float, metavar='weight',
            help="Weight of the scalar component (charges) in the model",
            required=True)
parser.add_argument(
    '-wt', '--vector-weight', type=float, metavar='weight',
            help="Weight of the vector component (point dipoles) in the model",
            required=True)
parser.add_argument(
    '-wp', '--write-properties', metavar='FILE', help="File in which to write "
            "the per-atom properties.  They are output in NumPy zipped format,"
            " with 'charges' containing the vector (N_atoms_total x 1) of "
            " per-atom charges and 'dipoles' containing the matrix "
            "(N_atoms_total x 3) of the Cartesian dipoles.")
parser.add_argument(
    '-wg', '--write-properties-geoms', metavar='FILE', help="Write the "
            "properties into the ASE Atoms list as tagged arrays associated to"
            " each geometry.")
parser.add_argument(
    '-tk', '--transpose-full-kernels', action='store_true', help="Transpose "
            "the full-sparse kernels, assuming they were stored in the "
            "opposite order (MN) from the one expected (NM) (where N is full "
            "and M is sparse)")
parser.add_argument(
    '-st', '--spherical-tensor-ordering', action='store_true',
           dest='spherical', help="Transform the vector kernels from spherical"
           " tensor to the internal Cartesian ordering")


def load_kernels(args):
    if args.scalar_weight != 0:
        scalar_kernel = np.load(args.scalar_kernel)
    else:
        scalar_kernel = np.array([])
    if args.vector_weight != 0:
        vector_kernel = np.load(args.vector_kernel)
    else:
        vector_kernel = np.array([])
    del args.scalar_kernel
    del args.vector_kernel
    return (scalar_kernel, vector_kernel)


if __name__ == "__main__":
    args = parser.parse_args()
    if ((args.write_properties is None)
            and (args.write_properties_geoms is None)):
        raise ValueError("You don't want to write out the properties, so "
                         "there's nothing for me to do.")
    scalar_kernel, vector_kernel = load_kernels(args)
    geometries = ase.io.read(args.geometries, ':')
    weights = np.load(args.weights)
    del args.weights
    compute_per_atom_properties(
            geometries, weights, scalar_kernel, vector_kernel,
            args.scalar_weight, args.vector_weight,
            args.transpose_full_kernels, args.spherical,
            args.write_properties, args.write_properties_geoms)

