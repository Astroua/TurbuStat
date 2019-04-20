
'''
Contains the code used in the tutorials. Saves the example images to the
images/ folder.
'''

import os
from os.path import join as osjoin
import matplotlib.pyplot as plt
import seaborn as sb
import astropy.units as u
from astropy.io import fits
import numpy as np
from spectral_cube import SpectralCube

# Use my default seaborn setting
sb.set(font='Sans-Serif', style='ticks')
sb.set_context("paper", font_scale=1.)

data_path = "../../../testingdata"

fig_path = 'images'

# Choose which methods to run
run_bispec = False
run_delvar = False
run_dendro = False
run_genus = False
run_mvc = False
run_pca = False
run_pdf = False
run_pspec = False
run_scf = False
run_moments = False
run_tsallis = False
run_vca = False
run_vcs = False
run_wavelet = False

# Bispectrum
if run_bispec:
    from turbustat.statistics import BiSpectrum

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]

    bispec = BiSpectrum(moment0)
    bispec.run(verbose=True, nsamples=10000,
               save_name=osjoin(fig_path, "bispectrum_design4.png"))

    # With mean subtraction
    bispec2 = BiSpectrum(moment0)
    bispec2.run(nsamples=10000, mean_subtract=True, seed=4242424)

    # Plot comparison w/ and w/o mean sub

    plt.subplot(121)
    plt.imshow(bispec.bicoherence, vmin=0, vmax=1, origin='lower')
    plt.title("Without mean subtraction")
    plt.subplot(122)
    plt.imshow(bispec2.bicoherence, vmin=0, vmax=1, origin='lower')
    plt.title("With mean subtraction")
    plt.savefig(osjoin(fig_path, "bispectrum_w_and_wo_meansub_coherence.png"))
    plt.close()

    # Radial and azimuthal slices
    rad_slices = bispec.radial_slice([30, 45, 60] * u.deg, 20 * u.deg, value='bispectrum_logamp')
    plt.errorbar(rad_slices[30][0], rad_slices[30][1], yerr=rad_slices[30][2], label='30')
    plt.errorbar(rad_slices[45][0], rad_slices[45][1], yerr=rad_slices[45][2], label='45')
    plt.errorbar(rad_slices[60][0], rad_slices[60][1], yerr=rad_slices[60][2], label='60')
    plt.legend()
    plt.xlabel("Radius")
    plt.ylabel("log Bispectrum")
    plt.grid()
    plt.tight_layout()
    plt.savefig(osjoin(fig_path, "bispectrum_radial_slices.png"))
    plt.close()

    azim_slices = bispec.azimuthal_slice([8, 16, 50], 10, value='bispectrum_logamp', bin_width=5 * u.deg)
    plt.errorbar(azim_slices[8][0], azim_slices[8][1], yerr=azim_slices[8][2], label='8')
    plt.errorbar(azim_slices[16][0], azim_slices[16][1], yerr=azim_slices[16][2], label='16')
    plt.errorbar(azim_slices[50][0], azim_slices[50][1], yerr=azim_slices[50][2], label='50')
    plt.legend()
    plt.xlabel("Theta (rad)")
    plt.ylabel("log Bispectrum")
    plt.grid()
    plt.tight_layout()
    plt.savefig(osjoin(fig_path, "bispectrum_azim_slices.png"))
    plt.close()


# Delta-Variance
if run_delvar:
    from turbustat.statistics import DeltaVariance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_err = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[1]

    delvar = DeltaVariance(moment0, weights=moment0_err, distance=250 * u.pc)
    delvar.run(verbose=True, xunit=u.pix,
               save_name=osjoin(fig_path, "delvar_design4.png"))

    # Now with fitting limits
    delvar.run(verbose=True, xunit=u.pix, xlow=4 * u.pix, xhigh=30 * u.pix,
               save_name=osjoin(fig_path, "delvar_design4_wlimits.png"))

    # Now with fitting limits
    delvar.run(verbose=True, xunit=u.pc, xlow=4 * u.pix, xhigh=30 * u.pix,
               save_name=osjoin(fig_path, "delvar_design4_physunits.png"))

    delvar.run(verbose=True, xunit=u.pc, xlow=4 * u.pix, xhigh=40 * u.pix,
               brk=8 * u.pix,
               save_name=osjoin(fig_path, "delvar_design4_break.png"))

    # Look at difference w/ non-periodic boundary handling
    # This needs to be revisited with the astropy convolution updates
    # delvar.run(verbose=True, xunit=u.pix, xlow=4 * u.pix, xhigh=30 * u.pix,
    #            boundary='fill',
    #            save_name=osjoin(fig_path, "delvar_design4_boundaryfill.png"))

# Dendrograms
if run_dendro:

    from turbustat.statistics import Dendrogram_Stats
    from astrodendro import Dendrogram

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]
    d = Dendrogram.compute(cube.data, min_value=0.005, min_delta=0.1, min_npix=50,
                           verbose=True)
    ax = plt.subplot(111)
    d.plotter().plot_tree(ax)
    plt.ylabel("Intensity (K)")
    plt.savefig(osjoin(fig_path, "design4_dendrogram.png"))
    plt.close()

    dend_stat = Dendrogram_Stats(cube, min_deltas=np.logspace(-2, 0, 50),
                                 dendro_params={"min_value": 0.005,
                                                "min_npix": 50})

    dend_stat.run(verbose=True,
                  save_name=osjoin(fig_path, "design4_dendrogram_stats.png"))

    # Periodic boundaries
    dend_stat.run(verbose=True, periodic_bounds=True,
                  save_name=osjoin(fig_path, "design4_dendrogram_stats_periodic.png"))

if run_genus:

    from turbustat.statistics import Genus

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]

    genus = Genus(moment0, lowdens_percent=15, highdens_percent=85, numpts=100,
                  smoothing_radii=np.linspace(1, moment0.shape[0] / 10., 5))
    genus.run(verbose=True, min_size=4, save_name=osjoin(fig_path, "genus_design4.png"))

    # With min/max values.
    genus = Genus(moment0, min_value=137, max_value=353, numpts=100,
                  smoothing_radii=np.linspace(1, moment0.shape[0] / 10., 5))
    genus.run(verbose=True, min_size=4, save_name=osjoin(fig_path, "genus_design4_minmaxval.png"))

    # Requiring regions be larger than the beam
    moment0.header["BMAJ"] = 2e-5  # deg.
    genus = Genus(moment0, lowdens_percent=15, highdens_percent=85,
                  smoothing_radii=[1] * u.pix)
    genus.run(verbose=True, use_beam=True, save_name=osjoin(fig_path, "genus_design4_beamarea.png"))

    # With a distance
    genus = Genus(moment0, lowdens_percent=15, highdens_percent=85,
                  smoothing_radii=u.Quantity([0.04 * u.pc]),
                  distance=250 * u.pc)
    genus.run(verbose=True, min_size=40 * u.AU**2,
              save_name=osjoin(fig_path, "genus_design4_physunits.png"))

# MVC
if run_mvc:

    from turbustat.statistics import MVC

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    centroid = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_centroid.fits"))[0]
    lwidth = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_linewidth.fits"))[0]

    mvc = MVC(centroid, moment0, lwidth)
    mvc.run(verbose=True, xunit=u.pix**-1,
            save_name=osjoin(fig_path, 'mvc_design4.png'))

    # With bounds
    mvc.run(verbose=True, xunit=u.pix**-1, low_cut=0.02 / u.pix,
            high_cut=0.1 / u.pix,
            save_name=osjoin(fig_path, 'mvc_design4_limitedfreq.png'))

    # With a break
    mvc = MVC(centroid, moment0, lwidth, distance=250 * u.pc)
    mvc.run(verbose=True, xunit=u.pix**-1, low_cut=0.02 / u.pix,
            high_cut=0.4 / u.pix,
            fit_kwargs=dict(brk=0.1 / u.pix), fit_2D=False,
            save_name=osjoin(fig_path, "mvc_design4_breakfit.png"))

    # With phys units
    mvc = MVC(centroid, moment0, lwidth, distance=250 * u.pc)
    mvc.run(verbose=True, xunit=u.pc**-1, low_cut=0.02 / u.pix,
            high_cut=0.1 / u.pix, fit_2D=False,
            save_name=osjoin(fig_path, 'mvc_design4_physunits.png'))

    # Azimuthal limits
    mvc = MVC(centroid, moment0, lwidth, distance=250 * u.pc)
    mvc.run(verbose=True, xunit=u.pc**-1, low_cut=0.02 / u.pix,
            high_cut=0.1 / u.pix, fit_2D=False,
            radial_pspec_kwargs={"theta_0": 1.13 * u.rad, "delta_theta": 40 * u.deg},
            save_name=osjoin(fig_path, 'mvc_design4_physunits_azimlimits.png'))

# PCA
if run_pca:

    from turbustat.statistics import PCA

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]

    pca = PCA(cube, distance=250. * u.pc)
    pca.run(verbose=True, mean_sub=False,
            min_eigval=1e-4, spatial_output_unit=u.pc,
            spectral_output_unit=u.m / u.s,
            beam_fwhm=10 * u.arcsec, brunt_beamcorrect=False,
            save_name=osjoin(fig_path, "pca_design4_default.png"))

    # With beam correction
    pca.run(verbose=True, mean_sub=False,
            min_eigval=1e-4, spatial_output_unit=u.pc,
            spectral_output_unit=u.m / u.s,
            beam_fwhm=10 * u.arcsec, brunt_beamcorrect=True,
            save_name=osjoin(fig_path, "pca_design4_beamcorr.png"))

    # With mean_sub
    pca_ms = PCA(cube, distance=250. * u.pc)
    pca_ms.run(verbose=True, mean_sub=True,
               min_eigval=1e-4, spatial_output_unit=u.pc,
               spectral_output_unit=u.m / u.s,
               beam_fwhm=10 * u.arcsec, brunt_beamcorrect=True,
               save_name=osjoin(fig_path, "pca_design4_meansub.png"))

    # Individual steps

    pca.compute_pca(mean_sub=False, n_eigs='auto', min_eigval=1.e-4, eigen_cut_method='value')
    print(pca.n_eigs)

    pca.compute_pca(mean_sub=False, n_eigs='auto', min_eigval=0.99, eigen_cut_method='proportion')
    print(pca.n_eigs)

    pca.compute_pca(mean_sub=False, n_eigs='auto', min_eigval=1.e-4, eigen_cut_method='value')
    pca.find_spatial_widths(method='contour', beam_fwhm=10 * u.arcsec,
                            brunt_beamcorrect=True, diagnosticplots=True)
    plt.savefig(osjoin(fig_path, "pca_autocorrimgs_contourfit_Design4.png"))
    plt.close()

    pca.find_spectral_widths(method='walk-down')
    autocorr_spec = pca.autocorr_spec()
    x = np.fft.rfftfreq(500) * 500
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 8))
    for i, ax in zip(range(9), axes.ravel()):
        ax.plot(x, autocorr_spec[:251, i])
        ax.axhline(np.exp(-1), label='exp(-1)', color='r', linestyle='--')
        ax.axvline(pca.spectral_width(u.pix)[i].value,
                   label='Fitted Width',
                   color='g', linestyle='-.')
        # ax.set_yticks([])
        ax.set_title("{}".format(i + 1))
        ax.set_xlim([0, 50])
        if i == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(osjoin(fig_path, "pca_autocorrspec_Design4.png"))
    plt.close()

    pca.fit_plaw(fit_method='odr', verbose=True)
    plt.savefig(osjoin(fig_path, "pca_design4_plaw_odr.png"))
    plt.close()

    pca.fit_plaw(fit_method='bayes', verbose=True)
    plt.savefig(osjoin(fig_path, "pca_design4_plaw_mcmc.png"))
    plt.close()

    print(pca.gamma)

    print(pca.sonic_length(T_k=10 * u.K, mu=1.36, unit=u.pc))

# PDF
if run_pdf:

    from turbustat.statistics import PDF

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    pdf_mom0 = PDF(moment0, min_val=0.0, bins=None)
    pdf_mom0.run(verbose=True,
                 save_name=osjoin(fig_path, "pdf_design4_mom0.png"))

    print(pdf_mom0.find_percentile(500))
    print(pdf_mom0.find_at_percentile(96.3134765625))

    moment0_error = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[1]
    pdf_mom0 = PDF(moment0, min_val=0.0, bins=None, weights=moment0_error.data**-2)
    pdf_mom0.run(verbose=True,
                 save_name=osjoin(fig_path, "pdf_design4_mom0_weights.png"))

    pdf_mom0 = PDF(moment0, normalization_type='standardize')
    pdf_mom0.run(verbose=True, do_fit=False,
                 save_name=osjoin(fig_path, "pdf_design4_mom0_stand.png"))

    pdf_mom0 = PDF(moment0, normalization_type='center')
    pdf_mom0.run(verbose=True, do_fit=False,
                 save_name=osjoin(fig_path, "pdf_design4_mom0_center.png"))

    pdf_mom0 = PDF(moment0, normalization_type='normalize')
    pdf_mom0.run(verbose=True, do_fit=False,
                 save_name=osjoin(fig_path, "pdf_design4_mom0_norm.png"))

    pdf_mom0 = PDF(moment0, normalization_type='normalize_by_mean')
    pdf_mom0.run(verbose=True, do_fit=False,
                 save_name=osjoin(fig_path, "pdf_design4_mom0_normmean.png"))

    pdf_mom0 = PDF(moment0, min_val=0.0, bins=None)
    pdf_mom0.run(verbose=True, fit_type='mcmc',
                 save_name=osjoin(fig_path, "pdf_design4_mom0_mcmc.png"))

    # Make a trace plot
    pdf_mom0.trace_plot()
    plt.savefig(osjoin(fig_path, "pdf_design4_mom0_mcmc_trace.png"))
    plt.close()

    # Make a corner plot
    pdf_mom0.corner_plot(quantiles=[0.16, 0.5, 0.84])
    plt.savefig(osjoin(fig_path, "pdf_design4_mom0_mcmc_corner.png"))
    plt.close()

    # Fit a power-law distribution
    import scipy.stats as stats
    pdf_mom0 = PDF(moment0, min_val=250.0, normalization_type=None)
    pdf_mom0.run(verbose=True, model=stats.pareto, fit_type='mle', floc=False,
                 save_name=osjoin(fig_path, "pdf_design4_mom0_plaw.png"))

    cube = SpectralCube.read(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))
    pdf_cube = PDF(cube)
    pdf_cube.run(verbose=True, do_fit=False,
                 save_name=osjoin(fig_path, "pdf_design4.png"))

# PSpec
if run_pspec:

    from turbustat.statistics import PowerSpectrum

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]

    pspec = PowerSpectrum(moment0, distance=250 * u.pc)
    pspec.run(verbose=True, xunit=u.pix**-1,
              save_name=osjoin(fig_path, "design4_pspec.png"))

    pspec.run(verbose=True, xunit=u.pix**-1,
              low_cut=0.025 / u.pix, high_cut=0.1 / u.pix,
              save_name=osjoin(fig_path, "design4_pspec_limitedfreq.png"))

    print(pspec.slope2D, pspec.slope2D_err)
    print(pspec.ellip2D, pspec.ellip2D_err)
    print(pspec.theta2D, pspec.theta2D_err)

    # How about fitting a break?
    pspec = PowerSpectrum(moment0, distance=250 * u.pc)
    pspec.run(verbose=True, xunit=u.pc**-1,
              low_cut=0.025 / u.pix, high_cut=0.4 / u.pix, fit_2D=False,
              fit_kwargs={'brk': 0.1 / u.pix, 'log_break': False},
              save_name=osjoin(fig_path, "design4_pspec_breakfit.png"))

    pspec = PowerSpectrum(moment0, distance=250 * u.pc)
    pspec.run(verbose=True, xunit=u.pc**-1,
              low_cut=0.025 / u.pix, high_cut=0.4 / u.pix, fit_2D=False,
              fit_kwargs={'brk': 0.1 / u.pix, 'log_break': False},
              radial_pspec_kwargs={"theta_0": 1.13 * u.rad, "delta_theta": 40 * u.deg},
              save_name=osjoin(fig_path, "design4_pspec_breakfit_azimlimits.png"))

    pspec = PowerSpectrum(moment0, distance=250 * u.pc)
    pspec.run(verbose=True, xunit=u.pix**-1,
              low_cut=0.025 / u.pix, high_cut=0.1 / u.pix,
              fit_kwargs={'weighted_fit': True},
              save_name=osjoin(fig_path, "design4_pspec_limitedfreq_weightfit.png"))


# SCF
if run_scf:

    from turbustat.statistics import SCF

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]

    scf = SCF(cube, size=11)
    scf.run(verbose=True,
            save_name=osjoin(fig_path, "design4_scf.png"))

    print(scf.slope2D, scf.slope2D_err)
    print(scf.ellip2D, scf.ellip2D_err)
    print(scf.theta2D, scf.theta2D_err)

    # With fit limits
    scf.run(verbose=True, xlow=1 * u.pix, xhigh=5 * u.pix,
            save_name=osjoin(fig_path, "design4_scf_fitlimits.png"))

    # With azimuthal constraints
    scf.run(verbose=True, xlow=1 * u.pix, xhigh=5 * u.pix,
            radialavg_kwargs={"theta_0": 1.13 * u.rad, "delta_theta": 70 * u.deg},
            save_name=osjoin(fig_path, "design4_scf_fitlimits_azimlimits.png"))


    # Custom lags w/ phys units
    distance = 250 * u.pc  # Assume a distance
    phys_conv = (np.abs(cube.header['CDELT2']) * u.deg).to(u.rad).value * distance
    custom_lags = np.arange(-4.5, 5, 1.5) * phys_conv
    scf_physroll = SCF(cube, roll_lags=custom_lags, distance=distance)
    scf_physroll.run(verbose=True, xunit=u.pc,
                     save_name=osjoin(fig_path, "design4_scf_physroll.png"))

    # boundary cut
    scf = SCF(cube, size=11)
    scf.run(verbose=True, boundary='cut',
            save_name=osjoin(fig_path, "design4_scf_boundcut.png"))


# Moments
if run_moments:

    from turbustat.statistics import StatMoments

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]

    moments = StatMoments(moment0, radius=5 * u.pix)
    moments.run(verbose=True, periodic=True,
                save_name=osjoin(fig_path, "design4_statmoments.png"))

    moments.plot_histograms(save_name=osjoin(fig_path, "design4_statmoments_hists.png"))

    # Demonstrate passing a weight array in
    np.random.seed(3434789)
    noise = moment0.data * 0.1 + np.random.normal(0, 0.1, size=moment0.data.shape)
    moments_weighted = StatMoments(moment0, radius=5 * u.pix, weights=noise**-2)
    moments_weighted.run(verbose=True, periodic=True,
                         save_name=osjoin(fig_path, "design4_statmoments_randweights.png"))
    moments_weighted.plot_histograms(save_name=osjoin(fig_path, "design4_statmoments_hists_randweights.png"))

    # Too small radius
    moments.run(verbose=False, radius=2 * u.pix)
    moments.plot_histograms(save_name=osjoin(fig_path, "design4_statmoments_hists_rad_2pix.png"))

    # Larger radius
    moments.run(verbose=False, radius=10 * u.pix)
    moments.plot_histograms(save_name=osjoin(fig_path, "design4_statmoments_hists_rad_10pix.png"))

    # Much larger radius
    moments.run(verbose=False, radius=32 * u.pix)
    moments.plot_histograms(save_name=osjoin(fig_path, "design4_statmoments_hists_rad_32pix.png"))

    # Other units
    moments = StatMoments(moment0, radius=0.25 * u.pc, distance=250 * u.pc)
    moments.run(verbose=False, periodic=True)
    moments.plot_histograms(save_name=osjoin(fig_path, "design4_statmoments_hists_physunits.png"))

# Tsallis
if run_tsallis:
    from turbustat.statistics import Tsallis

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]

    tsallis = Tsallis(moment0).run(verbose=True,
                                   save_name=osjoin(fig_path, 'design4_tsallis.png'))

    # Tsallis parameters plot.
    tsallis.plot_parameters(save_name=osjoin(fig_path, 'design4_tsallis_params.png'))

    # With physical lags units
    # Get a float rounding error, so just take first 3 decimal points
    phys_lags = np.around(np.arange(0.025, 0.5, 0.05), 3) * u.pc
    tsallis = Tsallis(moment0, lags=phys_lags, distance=250 * u.pc)
    tsallis.run(verbose=True,
                save_name=osjoin(fig_path, 'design4_tsallis_physlags.png'))

    # Not periodic
    tsallis_noper = Tsallis(moment0).run(verbose=True, periodic=False,
                                         save_name=osjoin(fig_path, 'design4_tsallis_noper.png'))

    # Change sigma clip
    tsallis = Tsallis(moment0).run(verbose=True, sigma_clip=3,
                                   save_name=osjoin(fig_path, 'design4_tsallis_sigclip.png'))

    tsallis.plot_parameters(save_name=osjoin(fig_path, 'design4_tsallis_params_sigclip.png'))

# VCA
if run_vca:

    from turbustat.statistics import VCA

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]

    vca = VCA(cube)
    vca.run(verbose=True,
            save_name=osjoin(fig_path, "design4_vca.png"))

    vca.run(verbose=True, xunit=u.pix**-1, low_cut=0.02 / u.pix, high_cut=0.1 / u.pix,
            save_name=osjoin(fig_path, "design4_vca_limitedfreq.png"))

    vca = VCA(cube, distance=250 * u.pc)
    vca.run(verbose=True, xunit=u.pc**-1, low_cut=0.02 / u.pix,
            high_cut=0.4 / u.pix,
            fit_kwargs=dict(brk=0.1 / u.pix), fit_2D=False,
            save_name=osjoin(fig_path, "design4_vca_breakfit.png"))

    vca_thicker = VCA(cube, distance=250 * u.pc, channel_width=400 * u.m / u.s)
    vca_thicker.run(verbose=True, xunit=u.pc**-1, low_cut=0.02 / u.pix,
                    high_cut=0.4 / u.pix,
                    fit_kwargs=dict(brk=0.1 / u.pix), fit_2D=False,
                    save_name=osjoin(fig_path, "design4_vca_400ms_channels.png"))

    # W/ azimuthal constraints
    vca = VCA(cube)
    vca.run(verbose=True, xunit=u.pix**-1, low_cut=0.02 / u.pix, high_cut=0.1 / u.pix,
            radial_pspec_kwargs={"theta_0": 1.13 * u.rad, "delta_theta": 40 * u.deg},
            save_name=osjoin(fig_path, "design4_vca_limitedfreq_azimilimits.png"))


# VCS
if run_vcs:

    from turbustat.statistics import VCS

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]

    vcs = VCS(cube)
    vcs.run(verbose=True,
            save_name=osjoin(fig_path, "design4_vcs.png"))

    vcs.run(verbose=True, high_cut=0.17 / u.pix,
            save_name=osjoin(fig_path, "design4_vcs_lowcut.png"))

    vcs.run(verbose=True, high_cut=0.17 / u.pix, xunit=(u.m / u.s)**-1,
            save_name=osjoin(fig_path, "design4_vcs_lowcut_physunits.png"))

    vcs.run(verbose=True, high_cut=0.17 / u.pix, low_cut=6e-4 / (u.m / u.s), xunit=(u.m / u.s)**-1,
            save_name=osjoin(fig_path, "design4_vcs_bothcut_physunits.png"))

# Wavelets
if run_wavelet:
    from turbustat.statistics import Wavelet

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]

    wavelet = Wavelet(moment0).run(verbose=True,
                                   save_name=osjoin(fig_path, 'design4_wavelet.png'))

    # Limit the range
    wavelet = Wavelet(moment0)
    wavelet.run(verbose=True, xlow=1 * u.pix, xhigh=10 * u.pix,
                save_name=osjoin(fig_path, 'design4_wavelet_fitlimits.png'))

    phys_scales = np.arange(0.025, 0.5, 0.05) * u.pc
    wavelet = Wavelet(moment0, distance=250 * u.pc, scales=phys_scales)
    wavelet.run(verbose=True, xlow=1 * u.pix, xhigh=10 * u.pix, xunit=u.pc,
                save_name=osjoin(fig_path, 'design4_wavelet_physunits.png'))

    wavelet = Wavelet(moment0)
    wavelet.run(verbose=True, scale_normalization=False, xhigh=10 * u.pix,
                save_name=osjoin(fig_path, 'design4_wavelet_unnorm.png'))
