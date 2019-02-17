
import os
from os.path import join as osjoin
import matplotlib.pyplot as plt
import seaborn as sb
import astropy.units as u
from astropy.io import fits
from spectral_cube import Projection
import numpy as np

from turbustat.simulator import make_extended

from turbustat.io.sim_tools import create_fits_hdu

from turbustat.statistics import PowerSpectrum

# Use my default seaborn setting
# sb.set(font='Times New Roman', style='ticks')
# sb.set_context("poster", font_scale=1.0,
#                rc={'figure.figsize': (11.7, 8.27)})

sb.set(font='Sans-Serif', style='ticks')
sb.set_context("paper", font_scale=1.)

# For some reason, the figure size isn't getting updated by the context
# change on my work desktop. Specify manually here
plt.rcParams['figure.figsize'] = (9, 6.3)

col_pal = sb.color_palette('colorblind')


fig_path = 'images'

# Choose which methods to run
run_apod_examples = False  # applying_apodizing_functions.rst
run_beamcorr_examples = False  # correcting_for_the_beam.rst
run_plawfield_examples = False  # generating_test_data.rst
run_missing_data_noise = False  # missing_data_noise.rst
run_data_for_tutorials = False  # data_for_tutorials.rst

if run_apod_examples:

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

    # Example of 2D Tukey power-spectrum
    plt.imshow(np.log10(np.fft.fftshift(np.abs(np.fft.fft2(data4))**2)))
    plt.savefig(osjoin(fig_path, '2d_tukey_pspec.png'))
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

    from radio_beam import Beam

    rnoise_img = make_extended(256, powerlaw=3.)

    pixel_scale = 3 * u.arcsec
    beamfwhm = 3 * u.arcsec
    imshape = rnoise_img.shape
    restfreq = 1.4 * u.GHz
    bunit = u.K

    plaw_hdu = create_fits_hdu(rnoise_img, pixel_scale, beamfwhm, imshape,
                               restfreq, bunit)

    pspec = PowerSpectrum(plaw_hdu)
    pspec.run(verbose=True, radial_pspec_kwargs={'binsize': 1.0},
              fit_kwargs={'weighted_fit': False}, fit_2D=False,
              low_cut=1. / (60 * u.pix),
              save_name=osjoin(fig_path, "rednoise_pspec_slope3.png"))

    pencil_beam = Beam(0 * u.deg)
    plaw_proj = Projection.from_hdu(plaw_hdu)
    plaw_proj = plaw_proj.with_beam(pencil_beam)

    new_beam = Beam(3 * plaw_hdu.header['CDELT2'] * u.deg)
    plaw_conv = plaw_proj.convolve_to(new_beam)

    plaw_conv.quicklook()
    plt.savefig('images/rednoise_slope3_img_smoothed.png')
    plt.close()

    pspec2 = PowerSpectrum(plaw_conv)
    pspec2.run(verbose=True, xunit=u.pix**-1, fit_2D=False,
               low_cut=0.025 / u.pix, high_cut=0.1 / u.pix,
               radial_pspec_kwargs={'binsize': 1.0},
               apodize_kernel='tukey')
    plt.axvline(np.log10(1 / 3.), color=col_pal[3], linewidth=8, alpha=0.8,
                zorder=1)
    plt.savefig("images/rednoise_pspec_slope3_smoothed.png")
    plt.close()

    pspec3 = PowerSpectrum(plaw_conv)
    pspec3.run(verbose=True, xunit=u.pix**-1, fit_2D=False,
               low_cut=0.025 / u.pix, high_cut=0.4 / u.pix,
               apodize_kernel='tukey', beam_correct=True)
    plt.axvline(np.log10(1 / 3.), color=col_pal[3], linewidth=8, alpha=0.8,
                zorder=1)
    plt.savefig("images/rednoise_pspec_slope3_smoothed_beamcorr.png")
    plt.close()

    # Overplot original, smooth, corrected, etc..

    pspec.plot_fit(color=col_pal[0], label='Original')
    pspec2.plot_fit(color=col_pal[1], label='Smoothed')
    pspec3.plot_fit(color=col_pal[2], label='Beam-Corrected')
    plt.legend(frameon=True, loc='lower left')
    plt.axvline(np.log10(1 / 3.), color=col_pal[3], linewidth=8, alpha=0.8, zorder=-1)
    plt.ylim([-2, 7.5])
    plt.tight_layout()
    plt.savefig("images/rednoise_pspec_slope3_beam_comparisons.png")
    plt.close()


if run_plawfield_examples:

    # Use the same settings as above for the isotropic case

    rnoise_img = make_extended(256, powerlaw=3.)

    pixel_scale = 3 * u.arcsec
    beamfwhm = 3 * u.arcsec
    imshape = rnoise_img.shape
    restfreq = 1.4 * u.GHz
    bunit = u.K

    # plaw_hdu = create_fits_hdu(rnoise_img, pixel_scale, beamfwhm, imshape,
    #                            restfreq, bunit)

    # pspec = PowerSpectrum(plaw_hdu)
    # pspec.run(verbose=True, radial_pspec_kwargs={'binsize': 1.0},
    #           fit_kwargs={'weighted_fit': False}, fit_2D=False,
    #           low_cut=1. / (60 * u.pix),
    #           save_name=osjoin(fig_path, "rednoise_pspec_slope3.png"))

    # 2D anisotropic

    rnoise_img = make_extended(256, powerlaw=3., ellip=0.5, theta=45 * u.deg)

    plt.imshow(rnoise_img)
    plt.savefig(osjoin(fig_path, "rednoise_slope3_ellip_05_theta_45.png"))
    plt.close()

    plaw_hdu = create_fits_hdu(rnoise_img, pixel_scale, beamfwhm, imshape,
                               restfreq, bunit)

    pspec = PowerSpectrum(plaw_hdu)
    pspec.run(verbose=True, radial_pspec_kwargs={'binsize': 1.0},
              fit_kwargs={'weighted_fit': False}, fit_2D=True,
              low_cut=1. / (60 * u.pix),
              save_name=osjoin(fig_path, "rednoise_pspec_slope3_ellip_05_theta_45.png"))

    # 3D

    from turbustat.simulator import make_3dfield

    threeD_field = make_3dfield(128, powerlaw=3.)

    plt.figure(figsize=[10, 3])
    plt.subplot(131)
    plt.imshow(threeD_field.mean(0), origin='lower')
    plt.subplot(132)
    plt.imshow(threeD_field.mean(1), origin='lower')
    plt.subplot(133)
    plt.imshow(threeD_field.mean(2), origin='lower')
    plt.tight_layout()
    plt.savefig(osjoin(fig_path, "rednoise_3D_slope3_projs.png"))
    plt.close()

    # PPV cube

    from turbustat.simulator import make_ppv

    velocity = make_3dfield(32, powerlaw=4., amp=1.,
                            randomseed=98734) * u.km / u.s

    # Deal with negative density values.
    density = make_3dfield(32, powerlaw=3., amp=1.,
                           randomseed=328764) * u.cm**-3
    density += density.std()
    density[density.value < 0.] = 0. * u.cm**-3

    T = 100 * u.K

    cube_hdu = make_ppv(velocity, density, los_axis=0,
                        vel_disp=np.std(velocity, axis=0).mean(),
                        T=T, chan_width=0.5 * u.km / u.s,
                        v_min=-20 * u.km / u.s, v_max=20 * u.km / u.s)

    from spectral_cube import SpectralCube

    cube = SpectralCube.read(cube_hdu)

    cube.moment0().quicklook()
    plt.colorbar()
    plt.savefig(osjoin(fig_path, "ppv_mom0.png"))
    plt.close()

    cube.moment1().quicklook()
    plt.colorbar()
    plt.savefig(osjoin(fig_path, "ppv_mom1.png"))
    plt.close()

    cube.mean(axis=(1, 2)).quicklook()
    plt.savefig(osjoin(fig_path, "ppv_mean_spec.png"))
    plt.close()

if run_missing_data_noise:

    from turbustat.statistics import PowerSpectrum, DeltaVariance

    img = make_extended(256, powerlaw=3., randomseed=54398493)

    # Now shuffle so the peak is near the centre
    img = np.roll(img, (128, -30), (0, 1))

    img -= img.min()

    plt.imshow(img, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(osjoin(fig_path, 'missing_data_image.png'))
    plt.close()

    # Show image in the tutorial

    pspec = PowerSpectrum(fits.PrimaryHDU(img))
    pspec.run(verbose=True)
    plt.savefig(osjoin(fig_path, 'missing_data_pspec.png'))
    plt.close()

    delvar = DeltaVariance(fits.PrimaryHDU(img))
    delvar.run(verbose=True)
    plt.savefig(osjoin(fig_path, 'missing_data_delvar.png'))
    plt.close()

    # Mask some of the data out

    masked_img = img.copy()
    masked_img[masked_img < np.percentile(img, 25)] = np.NaN

    plt.imshow(masked_img, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(osjoin(fig_path, 'missing_data_image_masked.png'))
    plt.close()

    pspec_masked = PowerSpectrum(fits.PrimaryHDU(masked_img))
    pspec_masked.run(verbose=True, high_cut=10**-1.25 / u.pix)
    plt.savefig(osjoin(fig_path, 'missing_data_pspec_masked.png'))
    plt.close()

    # Just masked
    delvar_masked = DeltaVariance(fits.PrimaryHDU(masked_img))
    delvar_masked.run(verbose=True, xlow=2 * u.pix, xhigh=50 * u.pix)
    plt.savefig(osjoin(fig_path, 'missing_data_delvar_masked.png'))
    plt.close()

    # What happens if we just pad that image with empty values

    # From:
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.pad.html
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0.)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    padded_masked_img = np.pad(masked_img, 128, pad_with, padder=np.NaN)

    from scipy import ndimage as nd
    labs, num = nd.label(np.isfinite(padded_masked_img), np.ones((3, 3)))
    # Keep the largest regions
    padded_masked_img[np.where(labs > 1)] = np.NaN

    plt.imshow(padded_masked_img, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(osjoin(fig_path, 'missing_data_image_masked_padded.png'))
    plt.close()

    pspec_masked_pad = PowerSpectrum(fits.PrimaryHDU(padded_masked_img))
    pspec_masked_pad.run(verbose=True, high_cut=10**-1.25 / u.pix)
    plt.savefig(osjoin(fig_path, 'missing_data_pspec_masked_pad.png'))
    plt.close()

    delvar_masked_padded = DeltaVariance(fits.PrimaryHDU(padded_masked_img))
    delvar_masked_padded.run(verbose=True, xlow=2 * u.pix, xhigh=70 * u.pix)
    plt.savefig(osjoin(fig_path, 'missing_data_delvar_masked_pad.png'))
    plt.close()

    noise_rms = 1.
    noisy_img = img + np.random.normal(0., noise_rms, img.shape)

    plt.imshow(noisy_img, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(osjoin(fig_path, 'missing_data_image_noisy.png'))
    plt.close()

    pspec_noisy = PowerSpectrum(fits.PrimaryHDU(noisy_img))
    pspec_noisy.run(verbose=True, high_cut=10**-1.2 / u.pix)
    plt.savefig(osjoin(fig_path, 'missing_data_pspec_noisy.png'))
    plt.close()

    delvar_noisy = DeltaVariance(fits.PrimaryHDU(noisy_img))
    delvar_noisy.run(verbose=True, xlow=10 * u.pix, xhigh=70 * u.pix)
    plt.savefig(osjoin(fig_path, 'missing_data_delvar_noisy.png'))
    plt.close()

    # noisy_masked_img = noisy_img.copy()
    # noisy_masked_img[noisy_masked_img < 5 * noise_rms] = np.NaN

    # pspec_noisy_masked = PowerSpectrum(fits.PrimaryHDU(noisy_masked_img))
    # pspec_noisy_masked.run(verbose=True, high_cut=10**-1.2 / u.pix)
    # plt.savefig(osjoin(fig_path, 'missing_data_pspec_noisy.png'))
    # plt.close()

    # delvar_noisy_masked = DeltaVariance(fits.PrimaryHDU(noisy_masked_img))
    # delvar_noisy_masked.run(verbose=True, xlow=10 * u.pix, xhigh=70 * u.pix)

if run_data_for_tutorials:

    # Make moment 0 maps for the tutorial data

    fid_mom0 = fits.open("../../testingdata/Fiducial0_flatrho_0021_00_radmc_moment0.fits")[0]
    des_mom0 = fits.open("../../testingdata/Design4_flatrho_0021_00_radmc_moment0.fits")[0]

    from mpl_toolkits.axes_grid import make_axes_locatable

    ax = plt.subplot(121)
    im1 = plt.imshow(fid_mom0.data / 1000., origin='lower')
    ax.set_title("Fiducial")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(im1, cax=cax)

    ax2 = plt.subplot(122)
    im2 = plt.imshow(des_mom0.data / 1000., origin='lower')
    ax2.set_title("Design")
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", "5%", pad="3%")
    cb2 = plt.colorbar(im2, cax=cax2)
    cb2.set_label(r"K km s$^{-1}$")

    plt.tight_layout()

    plt.savefig(osjoin(fig_path, "design_fiducial_moment0.png"))
    plt.close()
