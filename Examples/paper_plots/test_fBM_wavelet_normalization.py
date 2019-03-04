
'''
Make a plot of Wavelets with and without normalization

'''

# from turbustat.data_reduction import Mask_and_Moments
from turbustat.statistics import Wavelet
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

col_pal = sb.color_palette()

size = 256
slope = 3.

fig = plt.figure(figsize=figsize)

test_img = fits.PrimaryHDU(make_extended(size, powerlaw=slope))

unnorm_wave = Wavelet(test_img)
unnorm_wave.run(verbose=False,
                scale_normalization=False,
                xhigh=0.15 * size * u.pix,
                xlow=0.01 * size * u.pix)
unnorm_wave.plot_transform(show_residual=False, color=col_pal[0],
                           label='Unnormalized')

norm_wave = Wavelet(test_img)
norm_wave.run(verbose=False,
              scale_normalization=True,
              xhigh=0.15 * size * u.pix,
              xlow=0.01 * size * u.pix)
norm_wave.plot_transform(show_residual=False, color=col_pal[1],
                         symbol='o',
                         label='Normalized')
print(r"Slope: %0.2f+/-%0.2f" % (unnorm_wave.slope, unnorm_wave.slope_err))
print(r"Slope: %0.2f+/-%0.2f" % (norm_wave.slope, norm_wave.slope_err))

# plt.grid()
plt.ylim([0.1, 1e4])
plt.legend(frameon=True)

plt.savefig("../figures/wavelet_normalization.png")
plt.savefig("../figures/wavelet_normalization.pdf")
plt.close()
