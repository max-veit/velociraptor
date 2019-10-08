"""Interface to SOAPFAST to compute power spectra and kernels

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
logging.basicConfig()
LOGGER = logging.getLogger(__name__)


def compute_scalar_power_spectra(
        n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS0',
        atoms_file='qm7.xyz', workdir=None, n_sparse_envs=2000,
        n_sparse_components=500, feat_sparsefile=None):
    # compute powerspectra (soapfast)
    PY2_EXEC = os.environ.get('PY2_EXEC', 'python')
    SOAPFAST_PATH = os.environ.get('SOAPFAST_PATH', '')
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
        '-lm', '0', '-nc', str(n_sparse_components),
        '-rs', str(rad_c), str(rad_r0), str(rad_m),
        '-o', os.path.join(workdir, ps_prefix)
    ])
    if feat_sparsefile is not None:
        ps_args.extend(['-sf', os.path.join(workdir, feat_sparsefile)])
    LOGGER.info("Running: " + ' '.join(ps_args))
    subprocess.run(ps_args)

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
    subprocess.run(atomic_ps_args)

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
        subprocess.run(fps_args)


def compute_scalar_kernel(ps_name, ps_second_name=None, kernel_name='K0_MM',
                          zeta=2, workdir=None):
    PY2_EXEC = os.environ.get('PY2_EXEC', 'python')
    SOAPFAST_PATH = os.environ.get('SOAPFAST_PATH', '')
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
    subprocess.run(kernel_args)


def recompute_scalar_kernels(n_max, l_max, atom_width, rad_r0, rad_m,
                             n_sparse_envs=2000, n_sparse_components=500,
                             workdir=None):
    # number of sparse envs is another convergence parameter
    compute_scalar_power_spectra(
            n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS0_train',
            atoms_file='qm7_train.xyz', workdir=workdir,
            n_sparse_envs=n_sparse_envs,
            n_sparse_components=n_sparse_components)
    # Make sure feat sparsification uses the PS0_train values!
    # (and don't sparsify on envs)
    compute_scalar_power_spectra(
            n_max, l_max, atom_width, rad_r0, rad_m, ps_prefix='PS0_test',
            atoms_file='qm7_test.xyz', workdir=workdir,
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
    compute_scalar_kernel('PS0_test_atomic.npy', 'PS0_train_atomic_sparse.npy',
                          zeta=zeta, kernel_name='K0_TM', workdir=workdir)


#TODO this is a classic case of DRY -- it's the same function as in do_fit.py,
#     just in a different guise with different arguments (which are really just
#     presets).  Fix?
def load_train_kernels(workdir, weight_scalar, weight_tensor):
    if scalar_weight != 0.0:
        scalar_kernel_sparse = np.load(os.path.join(workdir, 'K0_MM.npy'))
        scalar_kernel_full_sparse = np.load(os.path.join(workdir, 'K0_NM.npy'))
    else:
        scalar_kernel_sparse = np.array([])
        scalar_kernel_full_sparse = np.array([])
    if tensor_weight != 0.0:
        tensor_kernel_sparse = np.load(os.path.join(workdir, 'Kvec_MM.npy'))
        tensor_kernel_full_sparse = np.load(os.path.join(workdir,
                                                         'Kvec_NM.npy'))
    else:
        tensor_kernel_sparse = np.array([])
        tensor_kernel_full_sparse = np.array([])
    return (scalar_kernel_sparse, scalar_kernel_full_sparse,
            tensor_kernel_sparse, tensor_kernel_full_sparse)


def load_test_kernels(workdir, weight_scalar, weight_tensor):
    if scalar_weight != 0.0:
        scalar_kernel_full_sparse = np.load(os.path.join(workdir, 'K0_TM.npy'))
    else:
        scalar_kernel_full_sparse = np.array([])
    if tensor_weight != 0.0:
        tensor_kernel_full_sparse = np.load(os.path.join(workdir,
                                                         'Kvec_TM.npy'))
    else:
        tensor_kernel_full_sparse = np.array([])
    return scalar_kernel_full_sparse, tensor_kernel_full_sparse


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


def compute_residual(n_max, l_max, atom_width, rad_r0, rad_m, dipole_reg,
                     charge_reg, weight_scalar=0.0, weight_tensor=0.0,
                     n_sparse_envs=2000, n_sparse_components=500, workdir=None,
                     dipole_normalize=True, recompute_kernels=True):

    if (weight_scalar == 0.0) and (weight_tensor == 0.0):
        raise ValueError("Can't have both scalar and tensor weights set "
                         "to zero")
    if recompute_kernels:
        if weight_tensor != 0.0:
            raise ValueError("Recomputing vector kernels is not yet supported")
        recompute_scalar_kernels(n_max, l_max, atom_width, rad_r0, rad_m,
                                 n_sparse_envs, n_sparse_components, workdir)

    # compute weights (velociraptor)
    dipoles_train = np.load(os.path.join(workdir, 'dipoles_train.npy'))
    geoms_train = ase.io.read(os.path.join(workdir, 'qm7_train.xyz'), ':')
    if dipole_normalize:
        natoms_train = np.array([geom.get_number_of_atoms()
                                 for geom in geoms_train])
        dipoles_train = dipoles_train / natoms_train[:, np.newaxis]
    charges_train = get_charges(geoms_train)
    (scalar_kernel_sparse, scalar_kernel_full_sparse,
     tensor_kernel_sparse, tensor_kernel_full_sparse) = load_train_kernels(
            workdir, weight_scalar, weight_tensor)
    (tensor_kernel_transposed,
     tensor_kernel_molecular) = infer_kernel_convention(
            tensor_kernel_full_sparse.shape, len(geoms_train), n_sparse_envs)
    scalar_kernel_transformed, tensor_kernel_transformed = transform_kernels(
            geoms_train, scalar_kernel_full_sparse, weight_scalar,
            tensor_kernel_full_sparse, weight_tensor,
            vector_kernel_molecular=tensor_kernel_molecular,
            transpose_scalar_kernel=False,
            transpose_vector_kernel=tensor_kernel_transposed)
    del scalar_kernel_full_sparse
    del tensor_kernel_full_sparse
    scalar_kernel_sparse, tensor_kernel_sparse = transform_sparse_kernels(
            geoms_train, scalar_kernel_sparse, weight_scalar,
            tensor_kernel_sparse, tensor_kernel)
    scalar_weights = compute_weights(
            dipoles_train, charges_train,
            scalar_kernel_sparse, scalar_kernel_transformed,
            tensor_kernel_sparse, tensor_kernel_transformed,
            scalar_weight=weight_scalar, tensor_weight=weight_tensor,
            charge_mode='fit',
            dipole_regularization=dipole_reg, charge_regularization=charge_reg,
            sparse_jitter=0.0)

    # compute residuals (velociraptor)
    dipoles_test = np.load(os.path.join(workdir, 'dipoles_test.npy'))
    geoms_test = ase.io.read(os.path.join(workdir, 'qm7_test.xyz'), ':')
    natoms_test = np.array([geom.get_number_of_atoms() for geom in geoms_test])
    if dipole_normalize:
        dipoles_test = dipoles_test / natoms_test[:, np.newaxis]
    charges_test = get_charges(geoms_test)
    scalar_kernel_test_train, tensor_kernel_test_train = load_test_kernels(
            workdir, weight_scalar, weight_tensor)
    (tensor_kernel_transposed,
     tensor_kernel_molecular) = infer_kernel_convention(
            tensor_kernel_test_train.shape, len(geoms_test), n_sparse_envs)
    (scalar_kernel_test_transformed,
     tensor_kernel_test_transformed) = transform_kernels(
            geoms_test, scalar_kernel_test_train, weight_scalar,
            tensor_kernel_test_train, weight_tensor,
            vector_kernel_molecular=tensor_kernel_molecular,
            transpose_scalar_kernel=False,
            transpose_vector_kernel=tensor_kernel_transposed)
    del scalar_kernel_test_train
    del tensor_kernel_test_train
    resids = compute_residuals(
            scalar_weights, dipoles_test, charges_test, natoms_test,
            scalar_kernel_test_transformed, tensor_kernel_test_transformed,
            scalar_weight=weight_scalar, tensor_weight=weight_tensor,
            charge_mode='fit', dipole_normalized=dipole_normalize)
    LOGGER.info("Finished computing test residual: {:.6g}".format(
        resids['dipole_rmse']))
    return resids['dipole_rmse']


