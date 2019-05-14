#!/usr/bin/python

import numpy as np

# Get spherical kernels
knm = np.load("K1_NM.npy")
kmm = np.load("K1_MM.npy")

# Create vector kernels
k_new_nm = np.zeros(np.shape(knm),dtype=float)
k_new_mm = np.zeros(np.shape(kmm),dtype=float)

# Transformation matrix
tmatr = np.array([[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0]])

# Fill kernels
for i in xrange(len(knm)):
    for j in xrange(len(knm[0])):
        k_new_nm[i,j] = np.dot(tmatr.T,np.dot(knm[i,j],tmatr))

for i in xrange(len(kmm)):
    for j in xrange(len(kmm[0])):
        k_new_mm[i,j] = np.dot(tmatr.T,np.dot(kmm[i,j],tmatr))

# Save vector kernels
np.save("Kvec_NM.npy",k_new_nm)
np.save("Kvec_MM.npy",k_new_mm)
