
'''
Same index and random seed as those use in the slope recovery script.
Larger size though: 256 vs 512 here.
'''

import numpy as np

from turbustat.simulator import make_extended

import astropy.io.fits as fits

import os

size = 512

path = os.path.expanduser("~/bigdata/ekoch/AstroStat/TurbuStat_Paper/fBM_images")

for slope in np.arange(0.5, 4.5, 0.5):
    test_img = fits.PrimaryHDU(make_extended(size, powerlaw=slope))

    imgname = "fBM_isotropic_image_slope_{}.fits".format(slope)

    test_img.writeto(os.path.join(path, imgname))
