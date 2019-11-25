#!/usr/bin/env python
"""Compute the cross-validation error of a given model on a dataset"""


import argparse
import logging
import os

import ase.io
import numpy as np
from scipy import optimize

from velociraptor.fitutils import (transform_kernels, transform_sparse_kernels,
                                   compute_weights, compute_residuals,
                                   get_charges)
from velociraptor.kerneltools import make_kernel_params
from velociraptor.kerneltools import compute_residual as kt_residual


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
            "more-or-less equal partitions and do cross validation with them",
            default=4)
parser.add_argument(
    '-cvf', '--cv-file', type=str, help="Use CV-indices from the given file, "
            "which must be a NumPy array with 'k' sets of indices over the "
            "first axis, denoting the points to be withheld as the test set. "
            "If the file does not exist, it will be created as described "
            "for 'k'.")
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
    '-wr', '--write-residuals', action='store_true', help="Write the "
            "individual (non-RMSed) residuals to a file called "
            "'cv_<n>_residuals.npz' in the working directory.")
#  parser.add_argument(
    #  '-wk', '--save-kernels', metavar='PREFIX', help="Save the kernels to files"
            #  " with the given prefix")
parser.add_argument(
    '-wd', '--working-directory', metavar='DIR', help="Working directory for "
            "power spectrum and kernel computations (geometry and dipole files"
            " are interpreted relative to this directory if paths are not "
            "otherwise specified", default='.')


def make_cv_sets(n_geoms, cv_num_partitions):
    idces_perm = np.random.permutation(n_geoms)
    idces_split = np.array_split(idces_perm, cv_num_partitions)
    return [np.sort(idces_set) for idces_set in idces_split]


def optimize_hypers(kparams_initial, dipole_reg_initial, charge_reg_initial,
                    scalar_weight, tensor_weight,
                    optimize_kparams, optimize_dreg, optimize_creg,
                    geometries, dipoles, workdir, cv_idces_sets,
                    dipole_normalize=True):
    if cv_idces_sets is None:
        LOGGER.error("Requested optimization with no cross-validation. "
                     "This is almost certainly a bad idea.")
    # Optimizing kernel params implies optimizing regularizers
    if optimize_kparams:
        def result_function(test_params):
            (dipole_reg_log, charge_reg_log,
             atom_width, rad_r0, rad_m) = test_params
            kparams = dict(kparams_initial)
            kparams['atom_width'] = atom_width
            kparams['rad_r0'] = rad_r0
            kparams['rad_m'] = rad_m
            return kt_residual(
                    10**dipole_reg_log, 10**charge_reg_log, scalar_weight,
                    tensor_weight, workdir, geometries, dipoles,
                    dipole_normalize, True, True, kparams, cv_idces_sets,
                    write_results=False, print_results=False)
        final_reg = np.array([dipole_reg_initial, charge_reg_initial])
        initial_params = np.concatenate((np.log10(final_reg),
                                        (kparams_initial['atom_width'],
                                         kparams_initial['rad_r0'],
                                         kparams_initial['rad_m'])))
        opt_result = optimize.minimize(result_function, initial_params,
                                       method='BFGS', options=dict(
                                           maxiter=100, disp=True))
        final_params = opt_result.x
        final_params[:2] = 10**final_params[:2]
        print(opt_result)
        print("Final parameters: " + np.array_str(final_params, precision=6))
        kparams_initial['atom_width'] = final_params[2]
        kparams_initial['rad_r0'] = final_params[3]
        kparams_initial['rad_m'] = final_params[4]
        final_reg = final_params[:2]
    elif optimize_dreg or optimize_creg:
        def result_function(dipole_reg_log, charge_reg_log):
            return kt_residual(10**dipole_reg_log, 10**charge_reg_log,
                               args.scalar_weight, args.tensor_weight,
                               args.working_directory, geometries, dipoles,
                               dipole_normalize, True, False, None,
                               cv_idces_sets, write_results=False,
                               print_results=False)
        final_reg = np.array([args.dipole_regularization,
                              args.charge_regularization])
        if not args.optimize_charge_reg:
            charge_reg_log = np.log10(args.charge_regularization)
            result_1d = lambda x: result_function(x, charge_reg_log)
            opt_result = optimize.minimize_scalar(result_1d)
            final_reg[0] = 10**opt_result.x
        elif not args.optimize_dipole_reg:
            dipole_reg_log = np.log10(args.dipole_regularization)
            result_1d = lambda x: result_function(dipole_reg_log, x)
            opt_result = optimize.minimize_scalar(result_1d)
            final_reg[1] = 10**opt_result.x
        else:
            result_2d = lambda x: result_function(*x)
            opt_result = optimize.minimize(
                    result_2d, (np.log10(args.dipole_regularization),
                                np.log10(args.charge_regularization)),
                    method='BFGS', options=dict(maxiter=100, disp=True))
            final_reg = 10**opt_result.x
        print(opt_result)
        print("Final regularizer: " + np.array_str(final_reg, precision=6))
    else:
        LOGGER.error("No optimization requested.")
        return None
    return kparams_initial, final_reg


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.dirname(args.geometries):
        geom_filename = os.path.join(args.working_directory, args.geometries)
    else:
        geom_filename = args.geometries
    del args.geometries
    if args.num_training_geometries:
        if len(args.num_training_geometries) > 1:
            raise NotImplementedError("Learning curves not yet implemented")
        else:
            n_train = args.num_training_geometries[0]
            geometries = ase.io.read(geom_filename, slice(0, n_train))
            fname_split = os.path.splitext(geom_filename)
            geom_filename = "{:s}_nt{:d}{:s}".format(
                    fname_split[0], n_train, fname_split[1])
            ase.io.write(geom_filename, geometries)
    else:
        geometries = ase.io.read(geom_filename, slice(None))
        n_train = len(geometries)
    charges = get_charges(geometries)
    dipole_fext = os.path.splitext(args.dipoles)[1]
    if not os.path.dirname(args.dipoles):
        dipole_filename = os.path.join(args.working_directory, args.dipoles)
    else:
        dipole_filename = args.dipoles
    del args.dipoles
    if dipole_fext == '.npy':
        dipoles = np.load(dipole_filename)[:n_train]
    elif (dipole_fext == '.txt') or (dipole_fext == '.dat'):
        dipoles = np.loadtxt(dipole_filename)[:n_train]
    else:
        logger.warning("Dipoles file has no filename extension; assuming "
                       "plain text.")
        dipoles = np.loadtxt(dipole_filename)[:n_train]
    natoms_list = [geom.get_number_of_atoms() for geom in geometries]
    dipole_normalize = True # seems to be the best option
    if args.cv_file is not None:
        try:
            cv_idces_sets = np.load(args.cv_file)
            if len(cv_idces_sets) != args.cv_num_partitions:
                LOGGER.warning(
                    "Different number of CV-partitions specified in the file "
                    "({:d}) and on the command line ({:d}); using the file "
                    "setting", len(cv_idces_sets), args.cv_num_partitions)
        except FileNotFoundError:
            LOGGER.info("CV-file not found, creating new CV-partition...")
            cv_idces_sets = make_cv_sets(n_train, args.cv_num_partitions)
            np.save(args.cv_file, cv_idces_sets)
    else:
        cv_idces_sets = make_cv_sets(n_train, args.cv_num_partitions)
    kparams = make_kernel_params(
        args.max_radial, args.max_angular, args.atom_sigma,
        args.radial_scaling_scale, args.radial_scaling_power, geom_filename,
        None, args.num_sparse_environments, args.num_sparse_features)
    result = kt_residual(
        args.dipole_regularization, args.charge_regularization,
        args.scalar_weight, args.tensor_weight, args.working_directory,
        geometries, dipoles,
        dipole_normalize, True, True, kparams, cv_idces_sets,
        args.write_residuals) # whew!
    if args.print_residuals:
        print("CV-error: {:.6f} a.u. per atom".format(result))
    if (args.optimize_charge_reg or args.optimize_dipole_reg
            or args.optimize_all):
        opt_results = optimize_hypers(
                kparams, args.dipole_regularization,
                args.charge_regularization,
                args.scalar_weight, args.tensor_weight,
                args.optimize_all, args.optimize_dipole_reg,
                args.optimize_charge_reg,
                geometries, dipoles, args.working_directory, cv_idces_sets,
                dipole_normalize)
        kparams_final, final_reg = opt_results
        # probably don't need to recompute kernel, since it'll be the last one
        # computed
        result = kt_residual(
            final_reg[0], final_reg[1],
            args.scalar_weight, args.tensor_weight, args.working_directory,
            geometries, dipoles,
            dipole_normalize, True, False, None, cv_idces_sets,
            args.write_residuals)
        if args.print_residuals:
            print("Final CV-error: {:.6f} a.u. per atom".format(result))
