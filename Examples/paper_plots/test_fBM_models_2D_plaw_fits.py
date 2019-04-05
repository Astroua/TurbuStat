
'''
Show off the 2D elliptical fits.
'''

import numpy as np

from turbustat.statistics import PowerSpectrum
from turbustat.simulator import make_extended
from turbustat.statistics.psds import make_radial_freq_arrays

import astropy.io.fits as fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u

plt.rcParams['axes.unicode_minus'] = False

size = 512
slope = 3

test_img = fits.PrimaryHDU(make_extended(size, powerlaw=slope,
                                         ellip=0.4,
                                         theta=(60 * u.deg).to(u.rad),
                                         randomseed=345987))

# The power-law behaviour continues up to ~1/4 of the size
pspec = PowerSpectrum(test_img)
pspec.run(fit_2D=True, radial_pspec_kwargs={'binsize': 2.0},
          fit_kwargs={'weighted_fit': False},
          low_cut=1 / (15 * u.pix),
          verbose=False)

print("{0}+/-{1}".format(pspec.slope, pspec.slope_err))
print("{0}+/-{1}".format(pspec.slope2D, pspec.slope2D_err))
print("{0}+/-{1}".format(pspec.ellip2D, pspec.ellip2D_err))
print("{0}+/-{1}".format(pspec.theta2D, pspec.theta2D_err))

# pspec.plot_fit(show_2D=True)

width = 8.75
# fig_ratio = (4.4 / 6.4) / 2
height = 5.07
figsize = (width, height)

fig = plt.figure(figsize=figsize)

ax1 = plt.subplot(121)
im1 = ax1.imshow(test_img.data.T, cmap='viridis', origin='lower')

divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("left", "5%", pad="3%")
cb = plt.colorbar(im1, cax=cax1)
cb.set_label(r"Image Value")
cax1.yaxis.set_ticks_position('left')
cax1.yaxis.set_label_position('left')

ax1.axes.xaxis.set_ticklabels([])
ax1.axes.yaxis.set_ticklabels([])

ax = plt.subplot(122)

yy_freq, xx_freq = make_radial_freq_arrays(pspec.ps2D.shape)

freqs_dist = np.sqrt(yy_freq**2 + xx_freq**2)

mask = np.logical_and(freqs_dist >= pspec.low_cut.value,
                      freqs_dist <= pspec.high_cut.value)

# Scale the colour map to be values within the mask
vmax = np.log10(pspec.ps2D[mask]).max()
vmin = np.log10(pspec.ps2D[mask]).min()

im2 = plt.imshow(np.log10(pspec.ps2D), interpolation="nearest",
                 origin="lower", vmax=vmax, vmin=vmin)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", "5%", pad="3%")
cb = plt.colorbar(im2, cax=cax)
cb.set_label(r"log $P_2 \ (K_x,\ K_y)$")

ax.contour(mask, colors=['r'], linestyles='--')

# Plot fit contours
ax.contour(pspec.fit2D(xx_freq, yy_freq), cmap='viridis')

ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.tight_layout()

plt.subplots_adjust(wspace=0.01)

plt.savefig("../figures/twod_field_anisotropy_plaw.png")
plt.savefig("../figures/twod_field_anisotropy_plaw.pdf")
plt.close()
