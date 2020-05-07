#!/usr/bin/env python2

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Convert spherical kernel to vector kernel",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-k", "--kernel", type=str, required=True, help="Kernel to convert")
parser.add_argument("-o", "--output", type=str, required=True, help="Output kernel name")
args = parser.parse_args()

# Get spherical kernels
k_in = np.load(args.kernel)

# Create vector kernels
k_out = np.zeros(np.shape(k_in),dtype=float)

# Transformation matrix
tmatr = np.array([[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0]])

# Fill kernels
for i in xrange(len(k_in)):
    for j in xrange(len(k_in[0])):
        k_out[i,j] = np.dot(tmatr.T,np.dot(k_in[i,j],tmatr))

# Save vector kernels
np.save(args.output,k_out)
