# VELOCIRAPTOR
Vector Learning Oggmented with Charges on Individual Realistic Atomic Positions to Optimize Regression

This is the code used to produce the results in: "Predicting molecular dipole
moments by combining atomic partial charges and atomic dipoles" (M. Veit,
D. M.  Wilkins, Y. Yang, R. A. DiStasio Jr., M. Ceriotti,
[arXiv:2003.12437](https://arxiv.org/abs/2003.12437) (2020)).


The first step in building a model is building the kernels, described below.
This step currently requires SOAPFAST, available here
([public version](https://github.com/dilkins/SOAPFAST-public), Python 2.7 only).
Follow the installation instructions there and make sure you set environment
variables properly to be able to call that code with the correct Python version.

## Command-line interface

VELOCIRAPTOR comes with several Python scripts to build kernels, to optimize
hyperparameters, and to fit and test models.  They are:

* `get_cv_error.py`     for computing kernels as well as optimizing kernel
                        hyperparameters and regularizers with respect to
                        cross-validation error (optimizing with respect to
                        non-CV training set error is a recipe for overfitting)
                        (Requires SOAPFAST)
* `do_fit.py`           for computing the weights of a fit once kernels have
                        been computed
* `get_residuals.py`    for computing the residuals of a fit on any new dataset
* `get_per_atom_predictions.py`
                        for decomposing the fitted model into per-atom
                        predictions (atomic charges, partial dipoles, or both)

Each script has a help message (run it with `-h`) that describes the available
options and behaviour.

## Python interface

There are two modules that expose a high-level interface, one for kernel
building (`kerneltools`) and one for fitting (`fitutils`).  See the module
docstrings for more information.  For examples of use, see the fit scripts,
particularly `get_cv_error.py` and `do_fit.py`.

## Building kernels, Bash style

The steps below describe how to build scalar and vector kernels directly using
the tools in SOAPFAST.  This may be useful if you want to do something not
wrapped in `kerneltools`.

To build scalar and vector kernels, first run the command:

    $ source env.sh

in velociraptor's base directory, to set environment variables, and then in a
folder containing the training structures as `train.xyz` run:

    $ bash get_training_kernels.sh -ne XXX

or

    $ bash get_training_kernels.sh -ne0 XXX -ne1 XXX

depending on whether you want to use the same environments for both (first case) or different environments (second case). Other useful flags are `-nc0 XXX` and `-nc1 XXX`, where in each case `XXX` is the number of features you want to keep in the L=0 or L=1 power spectrum building. *Probably vital is the flag `-ns XXX`*, which sets the number of frames you use to get the sparsification information for L=1 (for reference, on fidis I use 800). Finally, `-rc XXX` allows you to specify the radial cutoff.

This will create a folder, `PS_files`, which has all of the auxiliary information we will need later on (A matrices, list of environments, power spectra), and four kernels, `K0_NM.npy`, `K0_MM.npy`, `K1_NM.npy`, `K1_MM.npy`, needed for sparse GPR.

## License

VELOCIRAPTOR is licensed under the GPL, version 3; see the file LICENSE for
more details.  Copyright © 2020 Max Veit and David Wilkins.
