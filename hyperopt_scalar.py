#!/usr/bin/env python


import logging
import math
import os
import sys

import numpy as np
import scipy.optimize

from velociraptor.kerneltools import compute_residual

logging.basicConfig()
LOGGER = logging.getLogger(__name__)


def compute_cv_error_old(params, n_max, l_max, workdir, cv_basename='cv'):
    """This recomputes the kernels for each set, which we don't necessarily
    want (it should be fine to just use the same set of sparse points
    throughout).
    """
    atom_width, rad_r0, rad_m, dipole_reg_log, charge_reg_log = params
    dipole_reg = math.pow(10, dipole_reg_log)
    charge_reg = math.pow(10, charge_reg_log)
    rmse_sum = 0
    LOGGER.info("Trying params: " + str(params))
    with open(os.path.join(workdir, "opt_points.txt"), 'ab') as ptsf:
        np.savetxt(ptsf, params[np.newaxis, :])
    for cv_idx in range(4):
        workdir_cv = os.path.join(workdir, '{:s}_{:d}'.format(
            cv_basename, cv_idx))
        rmse_sum += compute_residual(
            n_max, l_max, atom_width, rad_r0, rad_m, dipole_reg, charge_reg,
            weight_scalar=1.0, weight_tensor=0.0,
            workdir=workdir_cv, recompute_kernels=True, dipole_normalize=False)
    LOGGER.info("CV residual is: {:.6f}".format(rmse_sum / 4.0))
    return rmse_sum / 4.0


def compute_cv_error(params, n_max, l_max, workdir, cv_basename='cv'):
    atom_width, rad_r0, rad_m, dipole_reg_log, charge_reg_log = params
    dipole_reg = math.pow(10, dipole_reg_log)
    charge_reg = math.pow(10, charge_reg_log)
    LOGGER.info("Trying params: " + str(params))
    with open(os.path.join(workdir, "opt_points.txt"), 'ab') as ptsf:
        np.savetxt(ptsf, params[np.newaxis, :])
    cv_idces_sets = []
    for cv_idx in range(4):
        cv_idces_sets.append(np.load(os.path.join(
            "{:s}_{:d}".format(cv_basename, cv_idx), 'idces.npy')))
    result = compute_residual(n_max, l_max, atom_width, rad_r0, rad_m,
            dipole_reg, charge_reg, weight_scalar=1.0, weight_tensor=0.0,
            workdir=workdir, n_sparse_envs=2000, recompute_kernels=True,
            dipole_normalize=False, cv_idces_sets=cv_idces_sets)
    LOGGER.info("CV residual is: {:.6f}".format(result))
    return result


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    #TODO do this properly with argparse at some point
    workdir = os.environ.get('WORKING_DIR', None)
    if workdir is None and (len(sys.argv) < 3):
        workdir = os.getcwd()
    else:
        workdir = sys.argv[2]
    opt_result = scipy.optimize.minimize(
        compute_cv_error, [0.5, 3.0, 2.0, -2, 2],
        args=(8, 6, workdir),
        method="Nelder-Mead",
        options={'initial_simplex': np.loadtxt(sys.argv[1])}
    )
    print("Success? {}".format(opt_result.success))
    print(opt_result.message)
    print(opt_result.x)
