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


def compute_cv_error(params, workdir=None, cv_basename='cv'):
    scalar_weight_log, tensor_weight_log, charge_reg_log = params
    weight_scalar = math.pow(10, scalar_weight_log)
    weight_tensor = math.pow(10, tensor_weight_log)
    charge_reg = math.pow(10, charge_reg_log)
    LOGGER.info("Trying params: " + str(params))
    with open(os.path.join(workdir, "opt_points.txt"), 'ab') as ptsf:
        np.savetxt(ptsf, params[np.newaxis, :])
    cv_idces_sets = np.load(os.path.join(workdir, "cv_idces.npy"))
    # The first few parameters are kernel parameters that don't matter since
    # we're not recomputing the kernel
    result = compute_residual(8, 6, 0.3, 2.5, 3.,
            1.0, charge_reg, weight_scalar, weight_tensor,
            workdir=workdir, n_sparse_envs=2000, recompute_kernels=False,
            dipole_normalize=True, cv_idces_sets=cv_idces_sets)
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
    result = scipy.optimize.minimize(
        compute_cv_error, [2.395046, 6.863596, 3.923451],
        args=(workdir,),
        method="Nelder-Mead",
        options={'initial_simplex': np.loadtxt(sys.argv[1])}
    )
    print("Success? {}".format(result.success))
    print(result.message)
    print(result.x)
