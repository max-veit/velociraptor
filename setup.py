import setuptools

setuptools.setup(name="velociraptor-dipoles", version="1.0",
                 packages=setuptools.find_packages(),
                 install_requires=["numpy>=1.14.0",
                                   "ase>=3.16.0"],
                 scripts=["do_fit.py",
                          "get_residuals.py",
                          "get_cv_error.py",
                          "get_per_atom_predictions.py"])
