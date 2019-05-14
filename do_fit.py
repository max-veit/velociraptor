"""Script to automate the process of fitting a model"""


import argparse
import sys

import ase

import fitutils
import transform


parser = argparse.ArgumentParser(
    description="Fit a model for the given set of dipoles")
parser.add_argument('geometries', help="Geometries of the molecules in the "
    "fit; should be the name of a file readable by ASE.")
parser.add_argument('dipoles', help="Dipoles, in Cartesian coordinates, "
    "per geometry.  Entries must be in the same order as the geometry file.")
parser.epilog("Charges are assumed to sum to zero for each geometry, unless "
    "the geometries have a property (info entry, in ASE terminology) named "
    "'total_charge'.\n\n"
    "Setting either of the (scalar or tensor) weights to zero will turn "
    "off that component completely and the corresponding kernel file(s) "
    "will not be read.")
parser.add_argument('scalar_kernel_sparse', help="Filename for the "
    "sparse-sparse (MM) scalar kernel, in atomic environment space")
parser.add_argument('scalar_kernel', help="Filename for the sparse-full "
    "(MN) scalar kernel, in atomic environment space")
parser.add_argument('tensor_kernel_sparse', help="Filename for the "
    "sparse-sparse tensor kernel")
parser.add_argument('tensor_kernel', help="Filename for the "
    "sparse-full tensor kernel, mapping environments to Cartesian components")
parser.add_argument('weights_output', help="Name of a file into which to "
     "write the output weights")
parser.add_argument('-ws', '--scalar-weight', type=float, help="Weight of "
    "the scalar component (charges) in the model", required=True)
parser.add_argument('-wt', '--tensor-weight', type=float, help="Weight of "
    "the tensor component (point dipoles) in the model", required=True)
parser.add_argument('-rc', '--charge-regularization', type=float, default=1.0,
                    help="Regularization coefficient (sigma) for total charges")
parser.add_argument('-rd', '--dipole-regularization', type=float, required=True,
                    help="Regularization coefficient (sigma) for "
                         "dipole components")
parser.add_argument('-sj', '--sparse-jitter', type=float, default=1E-8,
                    help="Small positive constant to ensure positive "
                    "definiteness of the kernel matrix
parser.add_argument('-m', '--charge-mode', choices=['none', 'fit', 'lagrange'],
                    help="How to control the total charge of each geometry. "
                    "Choices are 'none' (just fit dipoles), 'fit', (fit "
                    "dipoles and charges), and 'lagrange' (constrain total "
                    "charges exactly using Lagrange multipliers).",
                    default='fit')

def load_kernels(args):
    if args.scalar_weight != 0:
        scalar_kernel_sparse = np.load(args.scalar_kernel_sparse)
        scalar_kernel = np.load(args.scalar_kernel)
        scalar_kernel_sparse *= args.scalar_weight
        scalar_kernel *= args.scalar_weight
    else:
        scalar_kernel = np.array([])
        scalar_kernel_sparse = np.array([])
    if args.tensor_weight != 0:
        tensor_kernel_sparse = np.load(args.tensor_kernel_sparse)
        tensor_kernel = np.load(args.tensor_kernel)
        tensor_kernel_sparse *= args.tensor_weight
        tensor_kernel *= args.tensor_weight
    else:
        tensor_kernel = np.array([])
        tensor_kernel_sparse = np.array([])
    return (scalar_kernel_sparse, scalar_kernel,
            tensor_kernel_sparse, tensor_kernel)


def get_charges(geometries):
    return np.array([geom.info.get('total_charge', 0.) for geom in geometries])


if __name__ == "__main__":
    args = parser.parse_args(sys.argv)
    if args.charge_mode == 'none':
        raise ValueError('Bad idea')
    if (args.charge_mode == 'lagrange') and (args.tensor_weight != 0.):
        raise ValueError("Charge constraints not yet implemented together "
                         "with tensor fitting")
    kernels = load_kernels(args)
    geometries = ase.io.read(args.geometries)
    charges = get_charges(geometries)
    dipoles = np.loadtxt(args.dipoles)
    regularizer = fitutils.make_reg_vector(args.charge_regularization,
                                           args.dipole_regularization,
                                           len(geometries))
    scalar_kernel_transformed = transform.transform_envts_charge_dipoles(
            geometries, kernels[1].T)
    kernels_transformed = (kernels[0], scalar_kernel_transformed,
                           kernels[2], kernels[3].T) #TODO(max) check this
    if args.charge_mode == 'lagrange':
        #TODO(max) aaargh, this function takes descriptors not kernel matrices.
        #Fix it.
        weights = fitutils.compute_weights_charge_constrained(
                charges, dipoles, geometries,
                kernels[0], scalar_kernel_transformed, regularizer,
                sparse_jitter=args.sparse_jitter)
        np.save(args.weights_output, weights)
    elif args.charge_mode == 'fit':
        weights_combined = fitutils.compute_weights_two_model(
            charges, dipoles, geometries, regularizer, *kernels_transformed)
        np.save(args.weights_output, weights_combined)
