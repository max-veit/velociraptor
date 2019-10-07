#!/usr/bin/env python


import logging
import math
import os
import sys

import scipy.optimize

from velociraptor.kerneltools import compute_scalar_residual


def compute_cv_error(params, n_max, l_max, workdir=None, cv_basename='cv'):
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
    LOGGER.info("CV residual is: {:.4f}".format(rmse_sum / 4.0))
    return rmse_sum / 4.0


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    result = scipy.optimize.minimize(
        compute_cv_error, [0.5, 3.0, 2.0, -2, 2],
        args=(8, 6, "/scratch/veit/dipoles"),
        method="Nelder-Mead",
        options={'initial_simplex': np.loadtxt(sys.argv[1])}
    )
    print("Success? {}".format(result.success))
    print(result.message)
    print(result.x)
