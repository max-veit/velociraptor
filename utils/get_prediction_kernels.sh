#!/bin/bash

fname=showcase_dipoles.xyz   # File name of set whose properties we want to predict
rcut=4.0                     # Radial cutoff
suffix=TT                    # Suffix for kernel output files
envs=0                       # Do we use the same environments for both?

for arg in $(seq 1 $#);do
 if [ "${!arg}" == "-rc" ];then arg1=$((arg+1));rcut=${!arg1}
 elif [ "${!arg}" == "-f" ];then arg1=$((arg+1));fname=${!arg1}
 elif [ "${!arg}" == "-s" ];then arg1=$((arg+1));suffix=${!arg1}
 elif [ "${!arg}" == "-ne" ];then envs=1
 fi
done

mkdir prediction_files_${suffix}

# Get L=0 power spectrum
get_power_spectrum.py -rc ${rcut} -c H C N O S Cl -s H C N O S Cl -lm 0 -sf PS_files/PS0_train -f ${fname} -o prediction_files_${suffix}/PS0_pred

# Get L=1 power spectrum
get_power_spectrum.py -rc ${rcut} -c H C N O S Cl -s H C N O S Cl -lm 1 -sf PS_files/PS1_train -f ${fname} -o prediction_files_${suffix}/PS1_pred

# Get atomic power spectra
get_atomic_power_spectrum.py -lm 0 -p prediction_files_${suffix}/PS0_pred.npy -o prediction_files_${suffix}/PS0_pred_atomic -f ${fname}
get_atomic_power_spectrum.py -lm 1 -p prediction_files_${suffix}/PS1_pred.npy -o prediction_files_${suffix}/PS1_pred_atomic -f ${fname}

# Get prediction kernels
# L=0 kernel
get_kernel.py -lm 0 -z 2 -ps PS_files/PS0_train_atomic_sparse.npy prediction_files_${suffix}/PS0_pred_atomic.npy -ps0 PS_files/PS0_train_atomic_sparse.npy prediction_files_${suffix}/PS0_pred_atomic.npy -s NONE NONE -o K0_${suffix}

# L=1 kernel
if [ ${envs} == 1 ];then
 get_kernel.py -lm 1 -z 2 -ps PS_files/PS1_train_atomic_sparse.npy prediction_files_${suffix}/PS1_pred_atomic.npy -ps0 PS_files/PS0_train_atomic_sparse.npy prediction_files_${suffix}/PS0_pred_atomic.npy -s NONE NONE -o K1_${suffix}
else
 get_kernel.py -lm 1 -z 2 -ps PS_files/PS1_train_atomic_sparse.npy prediction_files_${suffix}/PS1_pred_atomic.npy -ps0 PS_files/PS01_train_atomic_sparse.npy prediction_files_${suffix}/PS0_pred_atomic.npy -s NONE NONE -o K1_${suffix}
fi

# Convert spherical kernel to vector kernel
spherical_to_cartesian_kernel.py -k K1_${suffix}.npy -o Kvec_${suffix}.npy

mv K1_${suffix}.npy prediction_files_${suffix}
