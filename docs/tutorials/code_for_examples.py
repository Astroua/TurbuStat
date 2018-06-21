
import os
from os.path import join as osjoin
import matplotlib.pyplot as plt
import seaborn as sb
import astropy.units as u
from astropy.io import fits
from spectral_cube import Projection
import numpy as np

from turbustat.tests.generate_test_images import make_extended
from turbustat.io.sim_tools import create_fits_hdu

# Use my default seaborn setting
sb.set(font='Times New Roman', style='ticks')
sb.set_context("poster", font_scale=1.0)


fig_path = 'images'

# Choose which methods to run
run_apod_examples = False
run_beamcorr_examples = False

if run_apod_examples:

    from turbustat.statistics import PowerSpectrum
    from turbustat.statistics.apodizing_kernels import \
        (CosineBellWindow, TukeyWindow, HanningWindow, SplitCosineBellWindow)

    # Apodizing kernel

    shape = (101, 101)
    taper = HanningWindow()
    data = taper(shape)

    plt.subplot(121)
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.subplot(122)
    plt.plot(data[shape[0] // 2])
    plt.savefig(osjoin(fig_path, 'hanning.png'))
    plt.close()

    taper2 = CosineBellWindow(alpha=0.98)
    data2 = taper2(shape)

    plt.subplot(121)
    plt.imshow(data2, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.subplot(122)
    plt.plot(data2[shape[0] // 2])
    plt.savefig(osjoin(fig_path, 'cosine.png'))
    plt.close()

    taper3 = SplitCosineBellWindow(alpha=0.3, beta=0.8)
    data3 = taper3(shape)

    plt.subplot(121)
    plt.imshow(data3, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.subplot(122)
    plt.plot(data3[shape[0] // 2])
    plt.savefig(osjoin(fig_path, 'splitcosine.png'))
    plt.close()

    taper4 = TukeyWindow(alpha=0.3)
    data4 = taper4(shape)

    plt.subplot(121)
    plt.imshow(data4, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.subplot(122)
    plt.plot(data4[shape[0] // 2])
    plt.savefig(osjoin(fig_path, 'tukey.png'))
    plt.close()

    plt.plot(data[shape[0] // 2], label='Hanning')
    plt.plot(data2[shape[0] // 2], label='Cosine')
    plt.plot(data3[shape[0] // 2], label='Split Cosine')
    plt.plot(data4[shape[0] // 2], label='Tukey')
    plt.legend(frameon=True)
    plt.savefig(osjoin(fig_path, '1d_apods.png'))
    plt.close()

    freqs = np.fft.rfftfreq(shape[0])
    plt.loglog(freqs, np.abs(np.fft.rfft(data[shape[0] // 2]))**2, label='Hanning')
    plt.loglog(freqs, np.abs(np.fft.rfft(data2[shape[0] // 2]))**2, label='Cosine')
    plt.loglog(freqs, np.abs(np.fft.rfft(data3[shape[0] // 2]))**2, label='Split Cosine')
    plt.loglog(freqs, np.abs(np.fft.rfft(data4[shape[0] // 2]))**2, label='Tukey')
    plt.legend(frameon=True)
    plt.xlabel("Freq. (1 / pix)")
    plt.ylabel("Power")
    plt.savefig(osjoin(fig_path, '1d_apods_pspec.png'))
    plt.close()

    # This is easier to show with a red noise image due to the limited
    # inertial range in the sim data.
    rnoise_img = make_extended(256, powerlaw=3.)

    pixel_scale = 3 * u.arcsec
    beamfwhm = 3 * u.arcsec
    imshape = rnoise_img.shape
    restfreq = 1.4 * u.GHz
    bunit = u.K

    plaw_hdu = create_fits_hdu(rnoise_img, pixel_scale, beamfwhm, imshape,
                               restfreq, bunit)

    plt.imshow(plaw_hdu.data, cmap='viridis')
    plt.savefig(osjoin(fig_path, "rednoise_slope3_img.png"))
    plt.close()

    pspec = PowerSpectrum(plaw_hdu)
    pspec.run(verbose=True, radial_pspec_kwargs={'binsize': 1.0},
              fit_kwargs={'weighted_fit': False}, fit_2D=False,
              low_cut=1. / (60 * u.pix),
              save_name=osjoin(fig_path, "rednoise_pspec_slope3.png"))

    pspec_partial = PowerSpectrum(rnoise_img[:128, :128], header=plaw_hdu.header)
    pspec_partial.run(verbose=False, fit_2D=False, low_cut=1 / (60. * u.pix))
    plt.imshow(np.log10(pspec_partial.ps2D))
    plt.savefig(osjoin(fig_path, "rednoise_pspec_slope3_2D_slicecross.png"))
    plt.close()

    pspec2 = PowerSpectrum(plaw_hdu)
    pspec2.run(verbose=False, radial_pspec_kwargs={'binsize': 1.0},
               fit_kwargs={'weighted_fit': False}, fit_2D=False,
               low_cut=1. / (60 * u.pix),
               apodize_kernel='hanning',)

    pspec3 = PowerSpectrum(plaw_hdu)
    pspec3.run(verbose=False, radial_pspec_kwargs={'binsize': 1.0},
               fit_kwargs={'weighted_fit': False}, fit_2D=False,
               low_cut=1. / (60 * u.pix),
               apodize_kernel='cosinebell', alpha=0.98,)

    pspec4 = PowerSpectrum(plaw_hdu)
    pspec4.run(verbose=False, radial_pspec_kwargs={'binsize': 1.0},
               fit_kwargs={'weighted_fit': False}, fit_2D=False,
               low_cut=1. / (60 * u.pix),
               apodize_kernel='splitcosinebell', alpha=0.3, beta=0.8)

    pspec5 = PowerSpectrum(plaw_hdu)
    pspec5.run(verbose=False, radial_pspec_kwargs={'binsize': 1.0},
               fit_kwargs={'weighted_fit': False}, fit_2D=False,
               low_cut=1. / (60 * u.pix),
               apodize_kernel='tukey', alpha=0.3)

    # Overplot all of the to demonstrate affect on large-scales.
    col_pal = sb.color_palette()

    pspec.plot_fit(color=col_pal[0], label='Original')
    pspec2.plot_fit(color=col_pal[1], label='Hanning')
    pspec3.plot_fit(color=col_pal[2], label='CosineBell')
    pspec4.plot_fit(color=col_pal[3], label='SplitCosineBell')
    pspec5.plot_fit(color=col_pal[4], label='Tukey')
    plt.legend(frameon=True, loc='lower left')
    plt.ylim([-2, 7.5])
    plt.tight_layout()
    plt.savefig(osjoin(fig_path, "rednoise_pspec_slope3_apod_comparisons.png"))
    plt.close()

    print("Original: {0:.2f} \nHanning: {1:.2f} \nCosineBell: {2:.2f} \n"
          "SplitCosineBell: {3:.2f} \nTukey: {4:.2f}"
          .format(pspec.slope, pspec2.slope, pspec3.slope, pspec4.slope,
                  pspec5.slope))

if run_beamcorr_examples:
    pass
