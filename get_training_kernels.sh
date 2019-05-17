#!/bin/bash

ne0=1000 # Number of environments for L=0
ne1=1000 # Number of environments for L=1
ne=-1    # Number of environments to be used for both
rcut=4.0 # Radial cutoff
nc0=600  # Feature sparsification for L=0
nc1=600  # Feature sparsification for L=1
ns=800   # Number of frames to use when sparsifying L=1 over features

for arg in $(seq 1 $#);do
 if [ "${!arg}" == "-rc" ];then arg1=$((arg+1));rcut=${!arg1}
 elif [ "${!arg}" == "-nc0" ];then arg1=$((arg+1));nc0=${!arg1}
 elif [ "${!arg}" == "-nc1" ];then arg1=$((arg+1));nc1=${!arg1}
 elif [ "${!arg}" == "-ne0" ];then arg1=$((arg+1));ne0=${!arg1}
 elif [ "${!arg}" == "-ne1" ];then arg1=$((arg+1));ne1=${!arg1}
 elif [ "${!arg}" == "-ns" ];then arg1=$((arg+1));ns=${!arg1}
 elif [ "${!arg}" == "-ne" ];then arg1=$((arg+1));ne=${!arg1}     # Specify this one if we want to use the same environments for both
 fi
done

mkdir PS_files

# Get L=0 power spectrum
get_power_spectrum.py -rc ${rcut} -c H C N O S Cl -s H C N O S Cl -lm 0 -nc ${nc0} -f train.xyz -o PS_files/PS0_train

# Get L=1 power spectrum
get_power_spectrum.py -rc ${rcut} -c H C N O S Cl -s H C N O S Cl -lm 1 -nc ${nc1} -f train.xyz -o PS_files/PS1_train -ns ${ns}
get_power_spectrum.py -rc 4.0 -c H C N O S Cl -s H C N O S Cl -lm 1 -f train.xyz -sf PS_files/PS1_train -o PS_files/PS1_train

# Find atomic power spectra and sparsify on environments
get_atomic_power_spectrum.py -lm 0 -p PS_files/PS0_train.npy -o PS_files/PS0_train_atomic -f train.xyz
get_atomic_power_spectrum.py -lm 1 -p PS_files/PS1_train.npy -o PS_files/PS1_train_atomic -f train.xyz
if [ ${ne} != -1 ];then
 # We want to use the same environments to sparsify on the L=0 and L=1
 # First get sparsification details
 do_fps.py -p PS_files/PS0_train_atomic.npy -n ${ne} -o PS_files/PS0_envs
 # Apply sparsification
 apply_fps.py -p PS_files/PS0_train_atomic.npy -sf PS_files/PS0_envs_rows -o PS_files/PS0_train_atomic_sparse
 apply_fps.py -p PS_files/PS1_train_atomic.npy -sf PS_files/PS0_envs_rows -o PS_files/PS1_train_atomic_sparse
else
 # We want to use different environments
 # First get sparsification details
 do_fps.py -p PS_files/PS0_train_atomic.npy -n ${ne0} -o PS_files/PS0_envs
 do_fps.py -p PS_files/PS1_train_atomic.npy -n ${ne1} -o PS_files/PS1_envs
 # Apply sparsification
 apply_fps.py -p PS_files/PS0_train_atomic.npy -sf PS_files/PS0_envs_rows -o PS_files/PS0_train_atomic_sparse
 apply_fps.py -p PS_files/PS1_train_atomic.npy -sf PS_files/PS1_envs_rows -o PS_files/PS1_train_atomic_sparse
 apply_fps.py -p PS_files/PS0_train_atomic.npy -sf PS_files/PS1_envs_rows -o PS_files/PS01_train_atomic_sparse
fi

# Generate sparsified kernels
if [ ${ne} != -1 ];then
 # Here we only have two power spectra, because we used the same environments for L=0 and L=1
 # Get L=0 kernel matrices
 get_kernel.py -lm 0 -z 2 -ps PS_files/PS0_train_atomic.npy PS_files/PS0_train_atomic_sparse.npy -ps0 PS_files/PS0_train_atomic.npy PS_files/PS0_train_atomic_sparse.npy -s NONE NONE -o K0_NM
 get_kernel.py -lm 0 -z 2 -ps PS_files/PS0_train_atomic_sparse.npy -ps0 PS_files/PS0_train_atomic_sparse.npy -s NONE NONE -o K0_MM
 # Get L=1 kernel matrices
 get_kernel.py -lm 1 -z 2 -ps PS_files/PS1_train_atomic.npy PS_files/PS1_train_atomic_sparse.npy -ps0 PS_files/PS0_train_atomic.npy PS_files/PS0_train_atomic_sparse.npy -s NONE NONE -o K1_NM
 get_kernel.py -lm 1 -z 2 -ps PS_files/PS1_train_atomic_sparse.npy -ps0 PS_files/PS0_train_atomic_sparse.npy -s NONE NONE -o K1_MM
else
 # We have three power spectra that must be used to build our sparsifieid kernels
 # Get L=0 kernel matrices
 get_kernel.py -lm 0 -z 2 -ps PS_files/PS0_train_atomic.npy PS_files/PS0_train_atomic_sparse.npy -ps0 PS_files/PS0_train_atomic.npy PS_files/PS0_train_atomic_sparse.npy -s NONE NONE -o K0_NM
 get_kernel.py -lm 0 -z 2 -ps PS_files/PS0_train_atomic_sparse.npy -ps0 PS_files/PS0_train_atomic_sparse.npy -s NONE NONE -o K0_MM
 # Get L=1 kernel matrices
 get_kernel.py -lm 1 -z 2 -ps PS_files/PS1_train_atomic.npy PS_files/PS1_train_atomic_sparse.npy -ps0 PS_files/PS0_train_atomic.npy PS_files/PS01_train_atomic_sparse.npy -s NONE NONE -o K1_NM
 get_kernel.py -lm 1 -z 2 -ps PS_files/PS1_train_atomic_sparse.npy -ps0 PS_files/PS01_train_atomic_sparse.npy -s NONE NONE -o K1_MM
fi

# Convert spherical kernels to vector kernels
spherical_to_cartesian_kernel.py -k K1_NM.npy -o Kvec_NM.npy
spherical_to_cartesian_kernel.py -k K1_MM.npy -o Kvec_MM.npy

mv K1_*.npy PS_files

# Print out hyperparameters
ofile=PS_files/kernel_hyperparameters.txt
echo  "Kernel hyperparameters"                 >  ${ofile}
echo  "======================"                 >> ${ofile}
echo                                           >> ${ofile}
echo  "Power Spectrum building:"               >> ${ofile}
echo  "nmax = 8"                               >> ${ofile}
echo  "lmax = 6"                               >> ${ofile}
echo  "radial cutoff = "${rcut}                >> ${ofile}
echo  "sigma = 0.3"                            >> ${ofile}
echo  "Chemical species = H C N O S Cl"        >> ${ofile}
echo  "Number of features kept (L=0) = "${nc0} >> ${ofile}
echo  "Number of features kept (L=1) = "${nc1} >> ${ofile}
if [ ${ne} != -1 ];then
 echo "Same environments used for both"        >> ${ofile}
 echo "Number of environments = "${ne}         >> ${ofile}
else
 echo "Different environments used"            >> ${ofile}
 echo "Number of environments (L=0) = "${ne0}  >> ${ofile}
 echo "Number of environments (L=1) = "${ne1}  >> ${ofile}
fi
echo                                           >> ${ofile}
echo "Kernel building:"                        >> ${ofile}
echo "zeta = 2"                                >> ${ofile}
