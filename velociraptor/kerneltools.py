"""Interface to SOAPFAST to compute power spectra and kernels

With some utilities for automating the fitting process for use in scripts

Since SOAPFAST is currently py2k-only, the interface needs to happen at
the process level; this is not much of a problem in practice.
"""


import logging
from math import sqrt
import os
import subprocess

import ase.io
import numpy as np

from .fitutils import (transform_kernels, transform_sparse_kernels,
                       compute_weights, compute_residuals, get_charges)

# PY2_EXEC = "/home/veit/miniconda3/envs/py2-compat/bin/python2"
LOGGER = logging.getLogger(__name__)
PY2_EXEC = os.environ.get('PY2_EXEC', 'python2.7')
PY2_DIR = os.environ.get('PY2_DIR', os.path.dirname(os.path.dirname(PY2_EXEC)))
PY2_ENV = dict(os.environ)
PY2_ENV['PATH'] = os.path.join(PY2_DIR, 'bin') + ':' + PY2_ENV.get('PATH', '')
PY2_ENV['PYTHONPATH'] = (
    os.path.join(PY2_DIR, 'lib', 'python2.7', 'site-packages') +
    ':' + PY2_ENV.get('PYTHONPATH', ''))
NCPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))


def compute_power_spectra(
        n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS', lambda_=0,
        atoms_file='qm7.xyz', workdir=None, n_sparse_envs=2000,
        n_sparse_components=500, feat_sparsefile=None, species_list=None):
    if species_list is None:
        species_list = 'H C N O S Cl'.split()
    # These PS parameters are unlikely to change
    rad_c = 1
    r_cut = 5.0
    if not os.path.dirname(atoms_file):
        atoms_file_rel = atoms_file
        atoms_file = os.path.join(workdir, atoms_file)
    else:
        atoms_file_rel = os.path.relpath(atoms_file, workdir)
    # This assumes SOAPFAST/bin is in PATH
    # Aaaand it needs SOAPFAST/scripts too.
    ps_args = ([
        'sagpr_parallel_get_PS',
        '-f', atoms_file_rel, '-nrun', str(NCPUS),
        '-lm', str(lambda_),
        '-n', str(n_max), '-l', str(l_max), '-rc', str(r_cut),
        '-sg', str(atom_width),
        '-c',] + species_list + ['-s',] + species_list + [
        '-rs', str(rad_c), str(rad_r0), str(rad_m),
        '-o', ps_prefix
    ])
    if feat_sparsefile is not None:
        ps_args.extend(['-sf', feat_sparsefile])
        ps_sparse_prefix = ps_prefix
    else:
        ps_args.extend(['-nc', str(n_sparse_components)])
        # The "sparse" suffix is only appended if creating a new feature FPS
        ps_sparse_prefix = ps_prefix + '_sparse'
    LOGGER.info("Running: " + ' '.join(ps_args))
    # This must be run in workdir because it creates tempfiles in its
    # current working directory
    subprocess.run(ps_args, env=PY2_ENV, cwd=workdir, check=True)

    # ATOMIC POWER spectra!
    atomic_ps_args = [
        'get_atomic_power_spectrum.py',
        '-p', os.path.join(workdir, ps_sparse_prefix + '.npy'),
        '-f', atoms_file,
        '-o', os.path.join(workdir, ps_prefix + '_atomic')
    ]
    LOGGER.info("Running: " + ' '.join(atomic_ps_args))
    subprocess.run(atomic_ps_args, env=PY2_ENV, check=True)

    # sparsify kernels (soapfast)
    if (n_sparse_envs is not None) and (n_sparse_envs > 0):
        fps_args = [
            'sagpr_do_env_fps',
            '-p', os.path.join(workdir, ps_prefix + '_atomic.npy'),
            #'-s', os.path.join(workdir, ps_prefix + '_natoms.npy'),
            '-n', str(n_sparse_envs),
            '-o', os.path.join(workdir, ps_prefix + '_atomic_fps'),
            '-a', os.path.join(workdir, ps_prefix + '_atomic_sparse')
        ]
        LOGGER.info("Running: " + ' '.join(fps_args))
        subprocess.run(fps_args, env=PY2_ENV, check=True)


def compute_scalar_power_spectra(
        n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS0',
        atoms_file='qm7.xyz', workdir=None, n_sparse_envs=2000,
        n_sparse_components=500, feat_sparsefile=None, species_list=None):
    lambda_ = 0
    compute_power_spectra(n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix,
                          lambda_, atoms_file, workdir, n_sparse_envs,
                          n_sparse_components, feat_sparsefile, species_list)


def compute_scalar_kernel(ps_name, ps_second_name=None, kernel_name='K0_MM',
                          zeta=2, workdir=None):
    if not os.path.dirname(ps_name):
        ps_name = os.path.join(workdir, ps_name)
    if (ps_second_name is not None) and not os.path.dirname(ps_second_name):
        ps_second_name = os.path.join(workdir, ps_second_name)
    # Warning: Only meant for lambda=0 PS for now, so PS and PS0 are identical
    if ps_second_name is not None:
        ps_files = [ps_name, ps_second_name]
    else:
        ps_files = [ps_name, ]
    kernel_args = ([
        'sagpr_get_kernel',
        '-z', str(zeta),
        '-ps', ] + ps_files + ['-ps0',] + ps_files + [
        '-o', os.path.join(workdir, kernel_name)
    ])
    LOGGER.info("Running: " + ' '.join(kernel_args))
    subprocess.run(kernel_args, env=PY2_ENV, check=True)


def compute_vector_kernel(ps_name, ps0_name, ps_second_name=None,
                          ps0_second_name=None, scaling_file=None,
                          scaling_file_second=None,
                          kernel_name='K1_MM',
                          zeta=2, workdir=None):
    #TODO this is getting ridiculous. Isn't there an os.path util for all these?
    if not os.path.dirname(ps_name):
        ps_name = os.path.join(workdir, ps_name)
    if not os.path.dirname(ps0_name):
        ps0_name = os.path.join(workdir, ps0_name)
    if (ps_second_name is not None) and not os.path.dirname(ps_second_name):
        ps_second_name = os.path.join(workdir, ps_second_name)
    if (ps0_second_name is not None) and not os.path.dirname(ps0_second_name):
        ps0_second_name = os.path.join(workdir, ps0_second_name)
    if (scaling_file is not None) and not os.path.dirname(scaling_file):
        scaling_file = os.path.join(workdir, scaling_file)
    if (ps_second_name is not None) and (ps0_second_name is None):
        raise ValueError("Must also provide a second lambda=0 powerspectrum if"
                         " providing a second lambda=1 powerspectrum")
    if ps_second_name is not None:
        ps_files = [ps_name, ps_second_name]
        ps0_files = [ps0_name, ps0_second_name]
    else:
        ps_files = [ps_name, ]
        ps0_files = [ps0_name, ]
    kernel_args = ([
        'sagpr_get_kernel',
        '-z', str(zeta),
        '-ps', ] + ps_files + ['-ps0',] + ps0_files + [
        '-o', os.path.join(workdir, kernel_name)
    ])
    if scaling_file is not None:
        if ps_second_name is None:
            kernel_args.extend(['-s', scaling_file])
        elif scaling_file_second is None:
            kernel_args.extend(['-s', scaling_file, 'NONE'])
        else:
            kernel_args.extend(['-s', scaling_file, scaling_file_second])
    elif (scaling_file_second is not None):
        kernel_args.extend(['-s', 'NONE', scaling_file_second])
    LOGGER.info("Running: " + ' '.join(kernel_args))
    subprocess.run(kernel_args, env=PY2_ENV, check=True)


def recompute_scalar_kernels(
        n_max, l_max, atom_width, rad_r0, rad_m,
        atoms_filename_train='qm7_train.xyz', atoms_filename_test=None,
        n_sparse_envs=2000, n_sparse_components=500, workdir=None,
        test_name='test', train_name='train',
        kernel_suffix='', species_list=None):
    if kernel_suffix:
        kernel_suffix = '_{:s}'.format(kernel_suffix)
    # number of sparse envs is another convergence parameter
    compute_scalar_power_spectra(
            n_max, l_max, atom_width, rad_r0, rad_m,
            ps_prefix='PS0_{:s}'.format(train_name),
            atoms_file=atoms_filename_train, workdir=workdir,
            n_sparse_envs=n_sparse_envs,
            n_sparse_components=n_sparse_components, species_list=species_list)
    # Make sure feat sparsification uses the PS0_train values!
    # (and don't sparsify on envs)
    if atoms_filename_test is not None:
        compute_scalar_power_spectra(
                n_max, l_max, atom_width, rad_r0, rad_m,
                ps_prefix='PS0_{:s}'.format(test_name),
                atoms_file=atoms_filename_test, workdir=workdir,
                feat_sparsefile='PS0_{:s}'.format(train_name),
                n_sparse_envs=-1, species_list=species_list)
    # compute kernels (soapfast)
    zeta = 2 # possibly another parameter to gridsearch
    # sparse-sparse
    compute_scalar_kernel('PS0_{:s}_atomic_sparse.npy'.format(train_name),
                          kernel_name=('K0_MM' + kernel_suffix), zeta=zeta,
                          workdir=workdir)
    # full-sparse
    compute_scalar_kernel(
            'PS0_{:s}_atomic.npy'.format(train_name),
            'PS0_{:s}_atomic_sparse.npy'.format(train_name), zeta=zeta,
            kernel_name=('K0_NM' + kernel_suffix), workdir=workdir)
    # test-train(sparse)
    if atoms_filename_test is not None:
        compute_scalar_kernel(
                'PS0_{:s}_atomic.npy'.format(test_name),
                'PS0_{:s}_atomic_sparse.npy'.format(train_name),
                zeta=zeta, kernel_name=('K0_TM' + kernel_suffix),
                workdir=workdir)


def recompute_vector_kernels(
        n_max, l_max, atom_width, rad_r0, rad_m,
        atoms_filename_train='qm7_train.xyz', atoms_filename_test=None,
        n_sparse_envs=2000, n_sparse_components=500, workdir=None,
        test_name='test', train_name='train', kernel_suffix='',
        species_list=None):
    if kernel_suffix:
        kernel_suffix = '_{:s}'.format(kernel_suffix)
    compute_power_spectra(
            n_max, l_max, atom_width, rad_r0, rad_m, lambda_=1,
            ps_prefix='PS1_{:s}'.format(train_name),
            atoms_file=atoms_filename_train, workdir=workdir,
            n_sparse_envs=n_sparse_envs,
            n_sparse_components=n_sparse_components, species_list=species_list)
    # Scalar power spectra are also needed for nonlinear vector kernels
    # Different name than for scalar kernels because the optimal
    # parameters for the two kernels will usually be different
    compute_power_spectra(
            n_max, l_max, atom_width, rad_r0, rad_m, lambda_=0,
            ps_prefix='PS0v_{:s}'.format(train_name),
            atoms_file=atoms_filename_train, workdir=workdir,
            n_sparse_envs=n_sparse_envs,
            n_sparse_components=n_sparse_components, species_list=species_list)
    # Make sure feat sparsification for test uses the train values!
    # (and don't sparsify on envs)
    if atoms_filename_test is not None:
        compute_power_spectra(
                n_max, l_max, atom_width, rad_r0, rad_m, lambda_=1,
                ps_prefix='PS1_{:s}'.format(test_name),
                atoms_file=atoms_filename_test, workdir=workdir,
                feat_sparsefile='PS1_{:s}'.format(train_name),
                n_sparse_envs=-1, species_list=species_list)
        compute_power_spectra(
                n_max, l_max, atom_width, rad_r0, rad_m, lambda_=0,
                ps_prefix='PS0v_{:s}'.format(test_name),
                atoms_file=atoms_filename_test, workdir=workdir,
                feat_sparsefile='PS0v_{:s}'.format(train_name),
                n_sparse_envs=-1, species_list=species_list)
    # compute kernels (soapfast)
    zeta = 2 # possibly another parameter to gridsearch
    # sparse-sparse
    compute_vector_kernel('PS1_{:s}_atomic_sparse.npy'.format(train_name),
                          'PS0v_{:s}_atomic_sparse.npy'.format(train_name),
                          kernel_name=('K1_MM' + kernel_suffix), zeta=zeta,
                          workdir=workdir)
    # full-sparse
    compute_vector_kernel(
            'PS1_{:s}_sparse.npy'.format(train_name),
            'PS0v_{:s}_sparse.npy'.format(train_name),
            'PS1_{:s}_atomic_sparse.npy'.format(train_name),
            'PS0v_{:s}_atomic_sparse.npy'.format(train_name),
            'PS1_{:s}_natoms.npy'.format(train_name),
            zeta=zeta, kernel_name=('K1_NM' + kernel_suffix), workdir=workdir)
    # test-train(sparse)
    if atoms_filename_test is not None:
        compute_vector_kernel(
                'PS1_{:s}.npy'.format(test_name),
                'PS0v_{:s}.npy'.format(test_name),
                'PS1_{:s}_atomic_sparse.npy'.format(train_name),
                'PS0v_{:s}_atomic_sparse.npy'.format(train_name),
                'PS1_{:s}_natoms.npy'.format(test_name),
                zeta=zeta, kernel_name=('K1_TM' + kernel_suffix),
                workdir=workdir)


#TODO this is a classic case of DRY -- it's the same function as in do_fit.py,
#     just in a different guise with different arguments (which are really just
#     presets).  Fix?
def load_kernels(workdir, weight_scalar, weight_vector, full_name='NM',
                 load_sparse=True, sparse_name='MM', spherical=False):
    if weight_scalar != 0.0:
        if load_sparse:
            scalar_kernel_sparse = np.load(
                    os.path.join(workdir, 'K0_{:s}.npy'.format(sparse_name)))
        scalar_kernel_full_sparse = np.load(
                os.path.join(workdir, 'K0_{:s}.npy'.format(full_name)))
    else:
        scalar_kernel_sparse = np.array([])
        scalar_kernel_full_sparse = np.array([])
    if weight_vector != 0.0:
        if spherical:
            vector_kernel_name = 'K1_{:s}.npy'
        else:
            vector_kernel_name = 'Kvec_{:s}.npy'
        if load_sparse:
            vector_kernel_sparse = np.load(
                    os.path.join(workdir,
                                 vector_kernel_name.format(sparse_name)))
        vector_kernel_full_sparse = np.load(
                os.path.join(workdir, vector_kernel_name.format(full_name)))
    else:
        vector_kernel_sparse = np.array([])
        vector_kernel_full_sparse = np.array([])
    if load_sparse:
        return (scalar_kernel_sparse, scalar_kernel_full_sparse,
                vector_kernel_sparse, vector_kernel_full_sparse)
    else:
        return scalar_kernel_full_sparse, vector_kernel_full_sparse


#TODO package the cv-split (or the data and kernels necessary for fitting, more
#     generally) into some sort of object
def do_cv_split(scalar_kernel_transformed, vector_kernel_transformed,
                geoms, dipoles, charges, idces_test):
    geoms_test = []
    geoms_train = []
    for idx, geom in enumerate(geoms):
        if idx in idces_test:
            geoms_test.append(geom)
        else:
            geoms_train.append(geom)
    idces_train = np.setdiff1d(np.arange(len(geoms)), idces_test, True)
    idces_all = np.arange(len(geoms)*4).reshape((-1, 4))
    idces_charge_dipole_test = idces_all[idces_test].flat
    idces_charge_dipole_train = idces_all[idces_train].flat
    if scalar_kernel_transformed.shape != (0,):
        scalar_kernel_test = scalar_kernel_transformed[
                idces_charge_dipole_test, :]
        scalar_kernel_train = scalar_kernel_transformed[
                idces_charge_dipole_train, :]
    else:
        scalar_kernel_test = scalar_kernel_train = scalar_kernel_transformed
    if vector_kernel_transformed.shape != (0,):
        vector_kernel_test = vector_kernel_transformed[
                idces_charge_dipole_test, :]
        vector_kernel_train = vector_kernel_transformed[
                idces_charge_dipole_train, :]
    else:
        vector_kernel_test = vector_kernel_train = vector_kernel_transformed
    dipoles_test = dipoles[idces_test, :]
    dipoles_train = dipoles[idces_train, :]
    charges_test = charges[idces_test]
    charges_train = charges[idces_train]
    return ((dipoles_test, charges_test, geoms_test,
             scalar_kernel_test, vector_kernel_test),
            (dipoles_train, charges_train, geoms_train,
             scalar_kernel_train, vector_kernel_train))


def infer_kernel_convention(kernel_shape, n_mol, n_atoms):
    # defaults
    transposed = False
    molecular = False
    if kernel_shape[0] == n_mol:
        LOGGER.info("Assuming kernel is molecular and not transposed.")
        molecular = True
    elif kernel_shape[0] == n_atoms:
        LOGGER.info("Assuming kernel is atomic and not transposed.")
    elif kernel_shape[1] == n_mol:
        LOGGER.info("Assuming kernel is molecular and transposed.")
        transposed = True
        molecular = True
    elif kernel_shape[1] == n_atoms:
        LOGGER.info("Assuming kernel is atomic and transposed.")
        transposed = True
    else:
        LOGGER.warn("Kernel shape: " + str(kernel_shape) + "unrecognized and "
                    "likely wrong! Using default settings, but expect shape "
                    "errors later on.")
    return transposed, molecular


def make_kernel_params(n_max, l_max, atom_width, rad_r0, rad_m,
                       atoms_filename_train, atoms_filename_test=None,
                       n_sparse_envs=2000, n_sparse_components=500):
    params = dict()
    params['n_max'] = n_max
    params['l_max'] = l_max
    params['atom_width'] = atom_width
    params['rad_r0'] = rad_r0
    params['rad_m'] = rad_m
    params['atoms_filename_train'] = atoms_filename_train
    params['atoms_filename_test'] = atoms_filename_test
    params['n_sparse_envs'] = n_sparse_envs
    params['n_sparse_components'] = n_sparse_components
    return params


def compute_residual(dipole_reg, charge_reg, weight_scalar, weight_vector,
                     workdir, geoms_train, dipoles_train,
                     dipole_normalize=True, spherical=False,
                     recompute_kernels=True, kparams=None, cv_idces_sets=None,
                     atoms_filename_test=None, dipoles_test=None,
                     write_results=True, print_results=True):
    if recompute_kernels:
        if cv_idces_sets is None:
            kparams['atoms_filename_test'] = 'qm7_test.xyz'
        else:
            kparams['atoms_filename_test'] = None
        if weight_scalar != 0.0:
            recompute_scalar_kernels(**kparams, workdir=workdir)
        if weight_vector != 0.0:
            recompute_vector_kernels(**kparams, workdir=workdir)
    if dipole_normalize:
        natoms_train = [len(geom) for geom in geoms_train]
        dipoles_train = (dipoles_train.T / natoms_train).T
    charges_train = get_charges(geoms_train)
    (scalar_kernel_sparse, scalar_kernel_transformed,
     vector_kernel_sparse, vector_kernel_transformed) = load_transform_kernels(
             workdir, geoms_train, weight_scalar, weight_vector,
             load_sparse=True, dipole_normalize=dipole_normalize,
             spherical=spherical)
    if cv_idces_sets is None:
        weights = compute_weights(
                dipoles_train, charges_train,
                scalar_kernel_sparse, scalar_kernel_transformed,
                vector_kernel_sparse, vector_kernel_transformed,
                scalar_weight=weight_scalar, vector_weight=weight_vector,
                charge_mode='fit', dipole_regularization=dipole_reg,
                charge_regularization=charge_reg, sparse_jitter=0.0)
        return compute_residual_from_weights(
                weights, weight_scalar, weight_vector, dipole_normalize,
                workdir, write_results, print_results)
    else:
        return compute_cv_residual(
                scalar_kernel_sparse, scalar_kernel_transformed,
                vector_kernel_sparse, vector_kernel_transformed,
                geoms_train, dipoles_train, charges_train,
                dipole_reg, charge_reg, weight_scalar, weight_vector,
                cv_idces_sets, workdir, dipole_normalize, write_results,
                print_results)


#TODO this is looking more and more like a constructor for a fitting class
#     that contains all the kernels and data necessary to do a fit
def load_transform_kernels(workdir, geoms, weight_scalar, weight_vector,
                           load_sparse=True, full_kernel_name='NM',
                           dipole_normalize=True, spherical=False):
    if (weight_scalar == 0.0) and (weight_vector == 0.0):
        raise ValueError("Can't have both scalar and vector weights set "
                         "to zero")
    if load_sparse:
        (scalar_kernel_sparse, scalar_kernel_full_sparse,
         vector_kernel_sparse, vector_kernel_full_sparse) = load_kernels(
                workdir, weight_scalar, weight_vector, load_sparse=True,
                full_name=full_kernel_name, spherical=spherical)
    else:
        scalar_kernel_full_sparse, vector_kernel_full_sparse = load_kernels(
                workdir, weight_scalar, weight_vector, load_sparse=False,
                full_name=full_kernel_name, spherical=spherical)
    #TODO if we generate the vector kernel ourselves, we don't have to
    #     infer the convention (though this should work just fine)
    if weight_vector != 0.0:
        (vector_kernel_transposed,
         vector_kernel_molecular) = infer_kernel_convention(
                vector_kernel_full_sparse.shape, len(geoms),
                sum(len(geom) for geom in geoms))
        if vector_kernel_transposed:
            raise ValueError("Transposed vector kernels are no longer "
                             "supported (re-generating is best)")
    else:
        vector_kernel_transposed = False
        vector_kernel_molecular = False
    # Assuming geometries was truncated using the ntrain option
    natoms_train = sum(len(geom) for geom in geoms)
    n_vector_rows = len(geoms) if vector_kernel_molecular else natoms_train
    scalar_kernel_transformed, vector_kernel_transformed = transform_kernels(
            geoms, scalar_kernel_full_sparse[:natoms_train], weight_scalar,
            vector_kernel_full_sparse[:n_vector_rows], weight_vector,
            vector_kernel_molecular=vector_kernel_molecular,
            transpose_full_kernels=False,
            dipole_normalize=dipole_normalize, spherical=spherical)
    del scalar_kernel_full_sparse
    del vector_kernel_full_sparse
    if load_sparse:
        scalar_kernel_sparse, vector_kernel_sparse = transform_sparse_kernels(
                geoms, scalar_kernel_sparse, weight_scalar,
                vector_kernel_sparse, weight_vector, spherical)
        return (scalar_kernel_sparse, scalar_kernel_transformed,
                vector_kernel_sparse, vector_kernel_transformed)
    else:
        return scalar_kernel_transformed, vector_kernel_transformed


def compute_cv_residual(
        scalar_kernel_sparse, scalar_kernel_transformed,
        vector_kernel_sparse, vector_kernel_transformed,
        geoms, dipoles, charges, dipole_reg, charge_reg,
        weight_scalar, weight_vector,
        cv_idces_sets, workdir, dipole_normalize=True,
        write_results=True, print_residuals=True):
    """Compute the CV residual by splitting the training data

    Uses kernels that have been pre-computed for the whole dataset
    """
    if (weight_scalar == 0.0) and (weight_vector == 0.0):
        raise ValueError("Can't have both scalar and vector weights set "
                         "to zero")
    rmse_sum = 0.0
    rmse_sum_sq = 0.0
    for cv_num, cv_idces in enumerate(cv_idces_sets):
        cv_test, cv_train = do_cv_split(scalar_kernel_transformed,
                                        vector_kernel_transformed,
                                        geoms, dipoles, charges, cv_idces)
        (dipoles_train, charges_train, geoms_train,
         scalar_kernel_train, vector_kernel_train) = cv_train
        weights = compute_weights(
                dipoles_train, charges_train, scalar_kernel_sparse,
                scalar_kernel_train, vector_kernel_sparse, vector_kernel_train,
                scalar_weight=weight_scalar, vector_weight=weight_vector,
                charge_mode='fit', dipole_regularization=dipole_reg,
                charge_regularization=charge_reg, sparse_jitter=0.0)
        if write_results:
            np.save(os.path.join(
                workdir, 'cv_{:d}_weights.npy'.format(cv_num)), weights)
        (dipoles_test, charges_test, geoms_test,
         scalar_kernel_test, vector_kernel_test) = cv_test
        natoms_test = [len(geom) for geom in geoms_test]
        write_residuals = (
                os.path.join(workdir, 'cv_{:d}_residuals.npz'.format(cv_num))
                if write_results else None)
        resid = compute_residuals(
                weights, dipoles_test, charges_test, natoms_test,
                scalar_kernel_test, vector_kernel_test,
                weight_scalar, weight_vector, charge_mode='fit',
                dipole_normalized=dipole_normalize,
                write_residuals=write_residuals,
                print_residuals=print_residuals)['dipole_rmse']
        LOGGER.debug("Finished computing test residual: {:.6g}".format(resid))
        rmse_sum += resid
        rmse_sum_sq += resid**2
    n_cv = len(cv_idces_sets)
    cv_error = rmse_sum / n_cv
    cv_stdev = sqrt(rmse_sum_sq/(n_cv - 1) - rmse_sum**2/(n_cv * (n_cv - 1)))
    LOGGER.info("CV-error:       {:.6g}".format(cv_error))
    LOGGER.info("CV-error stdev: {:.3g}".format(cv_stdev))
    return cv_error


def compute_residual_from_weights(weights, weight_scalar, weight_vector,
                                  dipole_normalize=True, workdir=None,
                                  write_results=True, print_residuals=True):
    dipoles_test = np.load(os.path.join(workdir, 'dipoles_test.npy'))
    geoms_test = ase.io.read(os.path.join(workdir, 'qm7_test.xyz'), ':')
    natoms_test = np.array([len(geom) for geom in geoms_test])
    charges_test = get_charges(geoms_test)
    if dipole_normalize:
        dipoles_test = dipoles_test / natoms_test[:, np.newaxis]
        charges_test = charges_test / natoms_test
    (scalar_kernel_test_transformed,
     vector_kernel_test_transformed) = load_transform_kernels(
         workdir, geoms_test, weight_scalar, weight_vector, load_sparse=False,
         full_kernel_name='TM', dipole_normalize=dipole_normalize)
    write_residuals = (os.path.join(os.path.join(workdir, 'test_residuals.npz')
                       if write_results else None))
    resids = compute_residuals(
            weights, dipoles_test, charges_test, natoms_test,
            scalar_kernel_test_transformed, vector_kernel_test_transformed,
            scalar_weight=weight_scalar, vector_weight=weight_vector,
            charge_mode='fit', dipole_normalized=dipole_normalize,
            write_residuals=write_residuals, print_residuals=print_residuals)
    LOGGER.info("Finished computing test residual: {:.6f}".format(
        resids['dipole_rmse']))
    return resids['dipole_rmse']

