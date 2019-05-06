"""Script to automate the process of fitting a model"""


import argparse
import sys

import fitutils


parser = argparse.ArgumentParser(
    description="Fit a model for the given set of dipoles")
parser.add_argument('geometries', help="Geometries of the molecules in the " +
        "fit; should be the name of a file readable by ASE.")
parser.add_argument('dipoles', help="Dipoles, in Cartesian coordinates, " +
        "per geometry.  Entries must be in the same order as the geometry " +
        "file.")
parser.epilog("Charges are assumed to sum to zero for each geometry, unless " +
        "the geometries have a property (info entry, in ASE terminology) " +
        "named 'total_charge'.\n\n"+
        "Setting either of the (scalar or tensor) weights to zero will turn " +
        "off that component completely and the corresponding kernel file " +
        "will not be read.")
parser.add_argument('scalar_kernel', help="Filename for the scalar "+
                        "kernel, in the format written by SOAPFATS")
#TODO how does soapfast store the sparse indices?
parser.add_argument('tensor_kernel', help="Filename for the l=1 " +
                        "kernel, in the format written by SOAPFATS")
parser.add_argument('weights_output', help="Name of a file into which to " +
     "write the output weights")
# The weights could also be incorporated directly into the kernel
parser.add_argument('-ws', '--scalar-weight', type=float, help="Weight of " +
    "the scalar component (charges) in the model")
parser.add_argument('-wt', '--tensor-weight', type=float, help="Weight of " +
    "the tensor component (point dipoles) in the model")
parser.add_argument('-rc', '--charge-regularization', type=float,
                    help="Regularization coefficient (sigma) for total charges")
parser.add_argument('-rd', '--dipole-regularization', type=float,
                    help="Regularization coefficient (sigma) for " +
                         "dipole components")
parser.add_argument('-m', '--charge-mode', choices=['none', 'fit', 'lagrange'],
                    help="How to control the total charge of each geometry. " +
                    "Choices are 'none' (just fit dipoles), 'fit', (fit " +
                    "dipoles and charges), and 'lagrange' (constrain total "+
                    "charges exactly using Lagrange multipliers).",
                    default='fit')

if __name__ == "__main__":
    args = parser.parse_args(sys.argv)
    weights_scalar, weights_tensor = fitutils.compute_weights_two_model()
    save_weights(args.weights_scalar, weights_tensor)
