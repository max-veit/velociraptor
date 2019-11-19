"""Interface to SOAPFAST to compute power spectra and kernels

With some utilities for automating the fitting process for use in scripts

Since SOAPFAST is currently py2k-only, the interface needs to happen at
the process level; this is not much of a problem in practice.
"""


import logging
import os
import subprocess

import ase.io
import numpy as np

from .fitutils import (transform_kernels, transform_sparse_kernels,
                       compute_weights, compute_residuals, get_charges)

# PY2_EXEC = "/home/veit/miniconda3/envs/py2-compat/bin/python2"
# SOAPFAST_PATH = "/home/veit/SOAPFAST/soapfast"
LOGGER = logging.getLogger(__name__)
PY2_EXEC = os.environ.get('PY2_EXEC', 'python')
SOAPFAST_PATH = os.environ.get('SOAPFAST_PATH', '')


def compute_power_spectra(
        n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS', lambda_=0,
        atoms_file='qm7.xyz', workdir=None, n_sparse_envs=2000,
        n_sparse_components=500, feat_sparsefile=None):
    # These PS parameters are unlikely to change
    rad_c = 1
    r_cut = 5.0
    if not os.path.dirname(atoms_file):
        atoms_file = os.path.join(workdir, atoms_file)
    ps_args = ([
        PY2_EXEC,
        os.path.join(SOAPFAST_PATH, 'get_power_spectrum.py'),
        '-f', atoms_file,
        '-n', str(n_max), '-l', str(l_max), '-rc', str(r_cut),
        '-sg', str(atom_width),
        '-c',] + 'H C N O S Cl'.split() + ['-s',] + 'H C N O S Cl'.split() + [
        '-lm', str(lambda_), '-nc', str(n_sparse_components),
        '-rs', str(rad_c), str(rad_r0), str(rad_m),
        '-o', os.path.join(workdir, ps_prefix)
    ])
    if feat_sparsefile is not None:
        ps_args.extend(['-sf', os.path.join(workdir, feat_sparsefile)])
    LOGGER.info("Running: ", *ps_args)
    subprocess.run(ps_args, check=True)

    # ATOMIC POWER spectra!
    atomic_ps_args = [
        PY2_EXEC,
        os.path.join(SOAPFAST_PATH, 'scripts',
                     'get_atomic_power_spectrum.py'),
        '-p', os.path.join(workdir, ps_prefix + '.npy'),
        '-f', atoms_file,
        '-o', os.path.join(workdir, ps_prefix + '_atomic')
    ]
    LOGGER.info("Running: " + ' '.join(atomic_ps_args))
    subprocess.run(atomic_ps_args, check=True)

    # sparsify kernels (soapfast)
    if (n_sparse_envs is not None) and (n_sparse_envs > 0):
        fps_args = [
            PY2_EXEC,
            os.path.join(SOAPFAST_PATH, 'scripts', 'do_fps.py'),
            '-p', os.path.join(workdir, ps_prefix + '_atomic.npy'),
            #'-s', os.path.join(workdir, ps_prefix + '_natoms.npy'),
            '-n', str(n_sparse_envs),
            '-o', os.path.join(workdir, ps_prefix + '_atomic_fps'),
            '-a', os.path.join(workdir, ps_prefix + '_atomic_sparse')
        ]
        LOGGER.info("Running: " + ' '.join(fps_args))
        subprocess.run(fps_args, check=True)


def compute_scalar_power_spectra(
        n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS0',
        atoms_file='qm7.xyz', workdir=None, n_sparse_envs=2000,
        n_sparse_components=500, feat_sparsefile=None):
    lambda_ = 0
    compute_power_spectra(n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix,
                          lambda_, atoms_file, workdir, n_sparse_envs,
                          n_sparse_components, feat_sparsefile)


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
        PY2_EXEC,
        os.path.join(SOAPFAST_PATH, 'get_kernel.py'),
        '-z', str(zeta),
        '-ps', ] + ps_files + ['-ps0',] + ps_files + [
        '-o', os.path.join(workdir, kernel_name)
    ])
    LOGGER.info("Running: " + ' '.join(kernel_args))
    subprocess.run(kernel_args, check=True)


def compute_vector_kernel(ps_name, ps0_name, ps_second_name=None,
                          ps0_second_name=None, kernel_name='K1_MM',
                          zeta=2, workdir=None):
    if not os.path.dirname(ps_name):
        ps_name = os.path.join(workdir, ps_name)
    if not os.path.dirname(ps0_name):
        ps_name = os.path.join(workdir, ps0_name)
    if (ps_second_name is not None) and not os.path.dirname(ps_second_name):
        ps_second_name = os.path.join(workdir, ps_second_name)
    if (ps0_second_name is not None) and not os.path.dirname(ps0_second_name):
        ps0_second_name = os.path.join(workdir, ps0_second_name)
    if (ps_second_name is not None) and (ps0_second_name is None):
        raise ValueError("Must also provide a second lambda=0 powerspectrum if"
                         " providing a second lambda=1 powerspectrum")
    if ps_second_name is not None:
        ps_files = [ps_name, ps_second_name]
        ps0_files = [ps0_name, ps0_second_name]
    else:
        ps_files = [ps_name, ]
    kernel_args = ([
        PY2_EXEC,
        os.path.join(SOAPFAST_PATH, 'get_kernel.py'),
        '-z', str(zeta),
        '-ps', ] + ps_files + ['-ps0',] + ps_files + [
        '-o', os.path.join(workdir, kernel_name)
    ])
    LOGGER.info("Running: " + ' '.join(kernel_args))
    subprocess.run(kernel_args, check=True)


def recompute_scalar_kernels(
        n_max, l_max, atom_width, rad_r0, rad_m,
        atoms_filename_train='qm7_train.xyz', atoms_filename_test=None,
        n_sparse_envs=2000, n_sparse_components=500, workdir=None):
    # number of sparse envs is another convergence parameter
    compute_scalar_power_spectra(
            n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS0_train',
            atoms_file=atoms_filename_train, workdir=workdir,
            n_sparse_envs=n_sparse_envs,
            n_sparse_components=n_sparse_components)
    # Make sure feat sparsification uses the PS0_train values!
    # (and don't sparsify on envs)
    if atoms_filename_test is not None:
        compute_scalar_power_spectra(
                n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS0_test',
                atoms_file=atoms_filename_test, workdir=workdir,
                feat_sparsefile='PS0_train', n_sparse_envs=-1)
    # compute kernels (soapfast)
    zeta = 2 # possibly another parameter to gridsearch
    # sparse-sparse
    compute_scalar_kernel('PS0_train_atomic_sparse.npy', kernel_name='K0_MM',
                          zeta=zeta, workdir=workdir)
    # full-sparse
    compute_scalar_kernel(
            'PS0_train_atomic.npy', 'PS0_train_atomic_sparse.npy', zeta=zeta,
            kernel_name='K0_NM', workdir=workdir)
    # test-train(sparse)
    if atoms_filename_test is not None:
        compute_scalar_kernel(
                'PS0_test_atomic.npy', 'PS0_train_atomic_sparse.npy',
                zeta=zeta, kernel_name='K0_TM', workdir=workdir)


def recompute_vector_kernels(
        n_max, l_max, atom_width, rad_r0, rad_m,
        atoms_filename_train='qm7_train.xyz', atoms_filename_test=None,
        n_sparse_envs=2000, n_sparse_components=500, workdir=None):
    compute_power_spectra(
            n_max, l_max, atom_width, rad_r0, rad_m, lambda_=1,
            ps_prefix='PS1_train',
            atoms_file=atoms_filename_train, workdir=workdir,
            n_sparse_envs=n_sparse_envs,
            n_sparse_components=n_sparse_components)
    # Make sure feat sparsification uses the PS1_train values!
    # (and don't sparsify on envs)
    if atoms_filename_test is not None:
        compute_power_spectra(
                n_max, l_max, atom_width, rad_r0, rad_m, lambda_=1,
                ps_prefix='PS1_test',
                atoms_file=atoms_filename_test, workdir=workdir,
                feat_sparsefile='PS1_train', n_sparse_envs=-1)
    # compute kernels (soapfast)
    zeta = 2 # possibly another parameter to gridsearch
    # sparse-sparse
    compute_vector_kernel('PS1_train_atomic_sparse.npy',
                          'PS0_train_atomic_sparse.npy', kernel_name='K1_MM',
                          zeta=zeta, workdir=workdir)
    # full-sparse
    compute_vector_kernel(
            'PS1_train.npy', 'PS0_train.npy',
            'PS1_train_atomic_sparse.npy', 'PS0_train_atomic_sparse.npy',
            zeta=zeta, kernel_name='K1_NM', workdir=workdir)
    # test-train(sparse)
    if atoms_filename_test is not None:
        compute_vector_kernel(
                'PS1_test.npy', 'PS0_test.npy',
                'PS1_train_atomic_sparse.npy', 'PS0_train_atomic_sparse.npy',
                zeta=zeta, kernel_name='K1_TM', workdir=workdir)
    #TODO transform spherical to Cartesian kernels!


#TODO this is a classic case of DRY -- it's the same function as in do_fit.py,
#     just in a different guise with different arguments (which are really just
#     presets).  Fix?
def load_kernels(workdir, weight_scalar, weight_tensor, full_name='NM',
                 load_sparse=True, sparse_name='MM'):
    if weight_scalar != 0.0:
        if load_sparse:
            scalar_kernel_sparse = np.load(
                    os.path.join(workdir, 'K0_{:s}.npy'.format(sparse_name)))
        scalar_kernel_full_sparse = np.load(
                os.path.join(workdir, 'K0_{:s}.npy'.format(full_name)))
    else:
        scalar_kernel_sparse = np.array([])
        scalar_kernel_full_sparse = np.array([])
    if weight_tensor != 0.0:
        if load_sparse:
            tensor_kernel_sparse = np.load(
                    os.path.join(workdir, 'Kvec_{:s}.npy'.format(sparse_name)))
        tensor_kernel_full_sparse = np.load(
                os.path.join(workdir, 'Kvec_{:s}.npy'.format(full_name)))
    else:
        tensor_kernel_sparse = np.array([])
        tensor_kernel_full_sparse = np.array([])
    if load_sparse:
        return (scalar_kernel_sparse, scalar_kernel_full_sparse,
                tensor_kernel_sparse, tensor_kernel_full_sparse)
    else:
        return scalar_kernel_full_sparse, tensor_kernel_full_sparse


#TODO package the cv-split (or the data and kernels necessary for fitting, more
#     generally) into some sort of object
def do_cv_split(scalar_kernel_transformed, tensor_kernel_transformed,
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
    if tensor_kernel_transformed.shape != (0,):
        tensor_kernel_test = tensor_kernel_transformed[
                idces_charge_dipole_test, :]
        tensor_kernel_train = tensor_kernel_transformed[
                idces_charge_dipole_train, :]
    else:
        tensor_kernel_test = tensor_kernel_train = tensor_kernel_transformed
    dipoles_test = dipoles[idces_test, :]
    dipoles_train = dipoles[idces_train, :]
    charges_test = charges[idces_test]
    charges_train = charges[idces_train]
    return ((dipoles_test, charges_test, geoms_test,
             scalar_kernel_test, tensor_kernel_test),
            (dipoles_train, charges_train, geoms_train,
             scalar_kernel_train, tensor_kernel_train))


def infer_kernel_convention(kernel_shape, n_mol, n_sparse_envs):
    # defaults
    transposed = False
    molecular = False
    if kernel_shape[0] == n_sparse_envs:
        if kernel_shape[1] == n_sparse_envs:
            LOGGER.warn(
                "Cannot infer whether kernel is transposed. (Are you sure "
                "this is the full-sparse and not the sparse-sparse kernel?) "
                "Using the default of False (kernel is NM-shaped).")
        elif kernel_shape[1] == n_mol:
            LOGGER.info("Assuming kernel is molecular and transposed.")
            transposed = True
            molecular = True
        else:
            LOGGER.info("Assuming kernel is atomic and transposed.")
            transposed = True
    elif kernel_shape[1] == n_sparse_envs:
        if kernel_shape[0] == n_mol:
            LOGGER.info("Assuming kernel is molecular and _not_ transposed.")
            molecular = True
        else:
            LOGGER.info("Assuming kernel is atomic and _not_ transposed.")
    else:
        LOGGER.warn("Kernel shape: " + str(kernel_shape) + "unrecognized and "
                    "likely wrong! Using default settings, but expect shape "
                    "errors later on.")
    return transposed, molecular


#TODO the kernel parameters aren't necessary if not recomputing the kernel,
#     might as well make a separate function (or make these optional, to keep
#     the interface)
def compute_residual(n_max, l_max, atom_width, rad_r0, rad_m, dipole_reg,
                     charge_reg, weight_scalar, weight_tensor, workdir,
                     n_sparse_envs=2000, n_sparse_components=500,
                     dipole_normalize=True, recompute_kernels=False,
                     cv_idces_sets=None):
    # hardcoded for now, sorry (not sorry)
    atoms_filename_train = 'qm7_train.xyz'
    dipoles_filename_train = 'dipoles_train.npy'
    if recompute_kernels:
        if weight_tensor != 0.0:
            raise ValueError("Recomputing vector kernels is not yet supported")
        if cv_idces_sets is None:
            atoms_filename_test = 'qm7_test.xyz'
        else:
            atoms_filename_test = None
        recompute_scalar_kernels(
                n_max, l_max, atom_width, rad_r0, rad_m,
                atoms_filename_train, atoms_filename_test,
                n_sparse_envs, n_sparse_components, workdir)
        #TODO recompute vector kernels, if _independently_ requested
    dipoles_train = np.load(os.path.join(workdir, dipoles_filename_train))
    geoms_train = ase.io.read(os.path.join(workdir, atoms_filename_train), ':')
    if dipole_normalize:
        natoms_train = np.array([geom.get_number_of_atoms()
                                 for geom in geoms_train])
        dipoles_train = dipoles_train / natoms_train[:, np.newaxis]
    charges_train = get_charges(geoms_train)
    (scalar_kernel_sparse, scalar_kernel_transformed,
     tensor_kernel_sparse, tensor_kernel_transformed) = load_transform_kernels(
             workdir, geoms_train, weight_scalar, weight_tensor,
             load_sparse=True, dipole_normalize=dipole_normalize)
    if cv_idces_sets is None:
        weights = compute_weights(
                dipoles_train, charges_train,
                scalar_kernel_sparse, scalar_kernel_transformed,
                tensor_kernel_sparse, tensor_kernel_transformed,
                scalar_weight=weight_scalar, tensor_weight=weight_tensor,
                charge_mode='fit', dipole_regularization=dipole_reg,
                charge_regularization=charge_reg, sparse_jitter=0.0)
        return compute_residual_from_weights(
                weights, weight_scalar, weight_tensor, dipole_normalize,
                workdir)
    else:
        return compute_cv_residual(
                scalar_kernel_sparse, scalar_kernel_transformed,
                tensor_kernel_sparse, tensor_kernel_transformed,
                geoms_train, dipoles_train, charges_train,
                dipole_reg, charge_reg, weight_scalar, weight_tensor,
                cv_idces_sets, workdir, dipole_normalize)


#TODO this is looking more and more like a constructor for a fitting class
#     that contains all the kernels and data necessary to do a fit
def load_transform_kernels(workdir, geoms, weight_scalar, weight_tensor,
                           load_sparse=True, full_kernel_name='NM',
                           dipole_normalize=True):
    if (weight_scalar == 0.0) and (weight_tensor == 0.0):
        raise ValueError("Can't have both scalar and tensor weights set "
                         "to zero")
    if load_sparse:
        (scalar_kernel_sparse, scalar_kernel_full_sparse,
         tensor_kernel_sparse, tensor_kernel_full_sparse) = load_kernels(
                workdir, weight_scalar, weight_tensor, load_sparse=True,
                full_name=full_kernel_name)
        n_sparse_envs = scalar_kernel_sparse.shape[0]
    else:
        scalar_kernel_full_sparse, tensor_kernel_full_sparse = load_kernels(
                workdir, weight_scalar, weight_tensor, load_sparse=False,
                full_name=full_kernel_name)
        # Assume the convention for the scalar kernels is _not_ transposed
        n_sparse_envs = scalar_kernel_full_sparse.shape[1]
    if weight_tensor != 0.0:
        (tensor_kernel_transposed,
         tensor_kernel_molecular) = infer_kernel_convention(
                tensor_kernel_full_sparse.shape, len(geoms), n_sparse_envs)
    else:
        tensor_kernel_transposed = False
        tensor_kernel_molecular = False
    scalar_kernel_transformed, tensor_kernel_transformed = transform_kernels(
            geoms, scalar_kernel_full_sparse, weight_scalar,
            tensor_kernel_full_sparse, weight_tensor,
            vector_kernel_molecular=tensor_kernel_molecular,
            transpose_scalar_kernel=False,
            transpose_vector_kernel=tensor_kernel_transposed,
            dipole_normalize=dipole_normalize)
    del scalar_kernel_full_sparse
    del tensor_kernel_full_sparse
    if load_sparse:
        scalar_kernel_sparse, tensor_kernel_sparse = transform_sparse_kernels(
                geoms, scalar_kernel_sparse, weight_scalar,
                tensor_kernel_sparse, weight_tensor)
        return (scalar_kernel_sparse, scalar_kernel_transformed,
                tensor_kernel_sparse, tensor_kernel_transformed)
    else:
        return scalar_kernel_transformed, tensor_kernel_transformed


def compute_cv_residual(
        scalar_kernel_sparse, scalar_kernel_transformed,
        tensor_kernel_sparse, tensor_kernel_transformed,
        geoms, dipoles, charges, dipole_reg, charge_reg,
        weight_scalar, weight_tensor,
        cv_idces_sets, workdir, dipole_normalize=True,
        write_results=True):
    """Compute the CV residual by splitting the training data

    Uses kernels that have been pre-computed for the whole dataset
    """
    if (weight_scalar == 0.0) and (weight_tensor == 0.0):
        raise ValueError("Can't have both scalar and tensor weights set "
                         "to zero")
    rmse_sum = 0.0
    for cv_num, cv_idces in enumerate(cv_idces_sets):
        cv_test, cv_train = do_cv_split(scalar_kernel_transformed,
                                        tensor_kernel_transformed,
                                        geoms, dipoles, charges, cv_idces)
        (dipoles_train, charges_train, geoms_train,
         scalar_kernel_train, tensor_kernel_train) = cv_train
        weights = compute_weights(
                dipoles_train, charges_train, scalar_kernel_sparse,
                scalar_kernel_train, tensor_kernel_sparse, tensor_kernel_train,
                scalar_weight=weight_scalar, tensor_weight=weight_tensor,
                charge_mode='fit', dipole_regularization=dipole_reg,
                charge_regularization=charge_reg, sparse_jitter=0.0)
        if write_results:
            np.save(os.path.join(
                workdir, 'cv_{:d}_weights.npz'.format(cv_num)), weights)
        (dipoles_test, charges_test, geoms_test,
         scalar_kernel_test, tensor_kernel_test) = cv_test
        natoms_test = [geom.get_number_of_atoms() for geom in geoms_test]
        write_residuals = (
                os.path.join(workdir, 'cv_{:d}_residuals.npz'.format(cv_num))
                if write_results else None)
        resid = compute_residuals(
                weights, dipoles_test, charges_test, natoms_test,
                scalar_kernel_test, tensor_kernel_test,
                weight_scalar, weight_tensor, charge_mode='fit',
                dipole_normalized=dipole_normalize,
                write_residuals=write_residuals)['dipole_rmse']
        LOGGER.info("Finished computing test residual: {:.6g}".format(resid))
        rmse_sum += resid
    return rmse_sum / len(cv_idces_sets)


def compute_residual_from_weights(weights, weight_scalar, weight_tensor,
                                  dipole_normalize=True, workdir=None):
    dipoles_test = np.load(os.path.join(workdir, 'dipoles_test.npy'))
    geoms_test = ase.io.read(os.path.join(workdir, 'qm7_test.xyz'), ':')
    natoms_test = np.array([geom.get_number_of_atoms() for geom in geoms_test])
    if dipole_normalize:
        dipoles_test = dipoles_test / natoms_test[:, np.newaxis]
    charges_test = get_charges(geoms_test)
    (scalar_kernel_test_transformed,
     tensor_kernel_test_transformed) = load_transform_kernels(
         workdir, geoms_test, weight_scalar, weight_tensor, load_sparse=False,
         full_kernel_name='TM', dipole_normalize=dipole_normalize)
    resids = compute_residuals(
            weights, dipoles_test, charges_test, natoms_test,
            scalar_kernel_test_transformed, tensor_kernel_test_transformed,
            scalar_weight=weight_scalar, tensor_weight=weight_tensor,
            charge_mode='fit', dipole_normalized=dipole_normalize)
    LOGGER.info("Finished computing test residual: {:.6g}".format(
        resids['dipole_rmse']))
    return resids['dipole_rmse']

