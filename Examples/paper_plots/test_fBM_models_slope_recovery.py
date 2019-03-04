
'''
How do the spectrum/log based statistics change using a power-law model image.

Compare:
    * power spectrum
    * delta-variance (slope + 2)
    * wavelets 2 * (slope + 1)

'''

import numpy as np

# from turbustat.data_reduction import Mask_and_Moments
from turbustat.statistics import DeltaVariance, PowerSpectrum, Wavelet
from turbustat.simulator import make_extended
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import astropy.units as u
import seaborn as sb

font_scale = 1.25

width = 4.2
# Keep the default ratio used in seaborn. This can get overwritten.
height = (4.4 / 6.4) * width
figsize = (width, height)

sb.set_context("paper", font_scale=font_scale,
               rc={"figure.figsize": figsize})
sb.set_palette("colorblind")

plt.rcParams['axes.unicode_minus'] = False

# Test the recovery of the index for delta-variance
slopes_wrap = []
slopes_pspec = []
slopes_wave = []

slopes_wrap_err = []
slopes_pspec_err = []
slopes_wave_err = []

size = 256

for slope in np.arange(0.5, 4.5, 0.5):
    test_img = fits.PrimaryHDU(make_extended(size, powerlaw=slope))
    # The power-law behaviour continues up to ~1/4 of the size
    delvar = DeltaVariance(test_img).run(xlow=3 * u.pix,
                                         xhigh=0.25 * size * u.pix,
                                         boundary='wrap')
    slopes_wrap.append(delvar.slope)
    slopes_wrap_err.append(delvar.slope_err)

    pspec = PowerSpectrum(test_img)
    pspec.run(fit_2D=False, radial_pspec_kwargs={'binsize': 2.0},
              fit_kwargs={'weighted_fit': False},
              low_cut=1 / (15 * u.pix),
              verbose=False)
    # plt.draw()
    # input("{0} {1}?".format(slope, - pspec.slope))
    # plt.clf()
    slopes_pspec.append(-pspec.slope)
    slopes_pspec_err.append(pspec.slope_err)

    wave = Wavelet(test_img).run(xhigh=0.15 * size * u.pix,
                                 xlow=0.02 * size * u.pix)
    slopes_wave.append(wave.slope)
    slopes_wave_err.append(wave.slope_err)

# 1.5 to ~3 are recovered

actual_slopes = np.arange(0.5, 4.5, 0.5)

plt.figure(figsize=figsize)

plt.axhline(0., linestyle='--', color='k', linewidth=4, alpha=0.7)
plt.errorbar(actual_slopes,
             ((np.array(slopes_wrap) + 2) - actual_slopes) / (actual_slopes * 0.01),
             yerr=slopes_wrap_err, label='Delta-Variance',
             marker='D')
plt.errorbar(actual_slopes,
             (np.array(slopes_pspec) - actual_slopes) / (actual_slopes * 0.01),
             yerr=slopes_pspec_err, label='Spatial Power Spectrum',
             marker='o')
plt.errorbar(actual_slopes,
             (2 * (np.array(slopes_wave) + 1) - actual_slopes) / (actual_slopes * 0.01),
             yerr=slopes_wave_err, label='Wavelet',
             marker='s')
# plt.plot(actual_slopes, actual_slopes, '--')
plt.legend(loc='best', frameon=True)
# plt.ylabel("Recovered Index")
plt.ylabel("Percent deviation in\nrecovered index")
plt.xlabel("Actual power-spectrum Index")
plt.grid()
plt.tight_layout()

plt.savefig("../figures/twod_field_index_recovery.png")
plt.savefig("../figures/twod_field_index_recovery.pdf")
plt.close()
