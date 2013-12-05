

'''

Perform a 2D Wavelet Transform on images

'''

# from kPyWavelet import twod as twodwt
from astropy.io.fits import getdata
import matplotlib.pyplot as p
import numpy as np

from wavelet_transform import *

img1, hdr1 = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.centroid.fits", header=True)
err_img1, err_hdr1 = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.centroid.error.fits", header=True)

img2, hdr2 = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.linewidth.fits", header=True)
err_img2, err_hdr2 = getdata("../../../../simcloud_stats/hd22_arrays/hd22.13co.linewidth.error.fits", header=True)

distance = 260 ## pc
imgscale1 = (hdr1['CDELT2']*(np.pi/180.0)*distance) ## pc/pix
imgscale2 = (hdr2['CDELT2']*(np.pi/180.0)*distance) ## pc/pix

a_min = round((5./3.),3) ## pix/pc
scales = np.logspace(np.log10(a_min), np.log10(min(img1.shape)/1.), 500)



test = Wavelet_Distance(img1 * err_img1**2., img2 * err_img2**2., imgscale1, imgscale2,scales=scales)
test.run()
p.plot(test.curve2[1],test.curve2[0],'r',test.curve1[1],test.curve1[0],'b')
print test.distance