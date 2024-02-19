import sys
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

file = sys.argv[1]

fid = open(file)

filesave = file[:-4]+"_smoothed300kms.txt"

fid2 = open(filesave,'w')

A = np.loadtxt(fid)

gauss_kernel1 = Gaussian1DKernel(1.28)  # Argument is sigma = FWHM/2.35. Model binning is 100 km/s, so 1.28 binning for sigma gives FWHM=300 km/s

y1 = convolve(A[:,1],gauss_kernel1)

np.savetxt(fid2, np.c_[A[:,0],y1],fmt='%12.4e')


