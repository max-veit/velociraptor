# VELOCIRAPTOR
Vector Learning Oggmented with Charges on Individual Realistic Atomic Positions

# To build kernels

To build scalar and vector kernels, first run the command:

::
  $ source env.sh
  
in GRUD's base directory, to set environment variables, and then in a folder containing `qm7b.xyz` run:

::
  $ bash get_training_kernels.sh -ne XXX
  
or

::
  $ bash get_training_kernels.sh -ne0 XXX -ne1 XXX

depending on whether you want to use the same environments for both (first case) or different environments (second case). Other useful flags are `-nc0 XXX` and `-nc1 XXX`, where in each case `XXX` is the number of features you want to keep in the L=0 or L=1 power spectrum building. *Probably vital is the flag `-ns XXX`*, which sets the number of frames you use to get the sparsification information for L=1 (for reference, on fidis I use 800). Finally, `-rc XXX` allows you to specify the radial cutoff.

This will create a folder, `PS_files`, which has all of the auxiliary information we will need later on (A matrices, list of environments, power spectra), and four kernels, `K0_NM.npy`, `K0_MM.npy`, `K1_NM.npy`, `K1_MM.npy`, needed for sparse GPR.
