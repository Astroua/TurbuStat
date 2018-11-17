
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
sb.set(font='Times New Roman', style='ticks')
sb.set_context("poster", font_scale=1.0)

data_path = "../../../testingdata"

fig_path = 'images'

# Choose which methods to run
run_bispec = False
run_cramer = False
run_delvar = False
run_dendro = False
run_genus = False
run_mvc = False
run_pca = False
run_pdf = False
run_pspec = False
run_scf = False
run_moments = False
# run_tsallis = False
run_vca = False
run_vcs = False
run_wavelet = False

# Bispectrum
if run_bispec:
    from turbustat.statistics import Bispectrum_Distance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[0]

    bispec = Bispectrum_Distance(moment0_fid, moment0,
                                 stat_kwargs={'nsamples': 10000})
    bispec.distance_metric(verbose=True,
                           save_name=osjoin(fig_path, "bispectrum_distmet.png"))

    print(bispec.surface_distance)
    print(bispec.mean_distance)

# Cramer Statistic
if run_cramer:

    from turbustat.statistics import Cramer_Distance

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]
    cube_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc.fits"))[0]

    cramer = Cramer_Distance(cube_fid, cube, noise_value1=-np.inf,
                             noise_value2=-np.inf)
    cramer.distance_metric(normalize=True, n_jobs=1, verbose=True,
                           save_name=osjoin(fig_path, "cramer_distmet.png"))

    print(cramer.distance)

# Delta-Variance
if run_delvar:
    from turbustat.statistics import DeltaVariance_Distance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_err = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[1]

    moment0_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_fid_err = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[1]

    delvar = DeltaVariance_Distance(moment0_fid, moment0, weights1=moment0_err,
                                    weights2=moment0_fid_err)
    delvar.distance_metric(verbose=True, xunit=u.pix,
                           save_name=osjoin(fig_path, "delvar_distmet.png"))
    print(delvar.curve_distance)
    print(delvar.slope_distance)

    # Now with fitting limits
    delvar_fit = DeltaVariance_Distance(moment0_fid, moment0, weights1=moment0_err,
                                        weights2=moment0_fid_err,
                                        delvar_kwargs={'xlow': 4 * u.pix,
                                                       'xhigh': 10 * u.pix})
    delvar_fit.distance_metric(verbose=True, xunit=u.pix,
                               save_name=osjoin(fig_path, "delvar_distmet_fitlimits.png"))

    print(delvar_fit.curve_distance)
    print(delvar_fit.slope_distance)

    delvar_fitdiff = DeltaVariance_Distance(moment0_fid, moment0, weights1=moment0_err,
                                            weights2=moment0_fid_err,
                                            delvar_kwargs={'xlow': 4 * u.pix,
                                                           'xhigh': 10 * u.pix},
                                            delvar2_kwargs={'xlow': 6 * u.pix,
                                                            'xhigh': 20 * u.pix})
    delvar_fitdiff.distance_metric(verbose=True, xunit=u.pix,
                                   save_name=osjoin(fig_path, "delvar_distmet_fitlimits_diff.png"))

    print(delvar_fitdiff.curve_distance)
    print(delvar_fitdiff.slope_distance)


# Dendrograms
if run_dendro:

    from turbustat.statistics import Dendrogram_Distance

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]
    cube_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc.fits"))[0]

    dend_dist = Dendrogram_Distance(cube_fid, cube,
                                    min_deltas=np.logspace(-2, 0, 50),
                                    dendro_params={"min_value": 0.005,
                                                   "min_npix": 50})

    dend_dist.distance_metric(verbose=True,
                  save_name=osjoin(fig_path, "dendrogram_distmet.png"))

    print(dend_dist.histogram_distance)
    print(dend_dist.num_distance)

if run_genus:

    from turbustat.statistics import Genus_Distance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[0]

    genus = Genus_Distance(moment0_fid, moment0,
                           lowdens_percent=15, highdens_percent=85, numpts=100,
                           genus_kwargs=dict(min_size=4 * u.pix**2),
                           smoothing_radii=np.linspace(1, moment0.shape[0] / 10., 5))
    genus.distance_metric(verbose=True, save_name=osjoin(fig_path, "genus_distmet.png"))

# MVC
if run_mvc:

    from turbustat.statistics import MVC_Distance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    centroid = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_centroid.fits"))[0]
    lwidth = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_linewidth.fits"))[0]

    data = {"moment0": [moment0.data, moment0.header],
            "centroid": [centroid.data, centroid.header],
            "linewidth": [lwidth.data, lwidth.header]}

    moment0_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[0]
    centroid_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_centroid.fits"))[0]
    lwidth_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_linewidth.fits"))[0]

    data_fid = {"moment0": [moment0_fid.data, moment0_fid.header],
                "centroid": [centroid_fid.data, centroid_fid.header],
                "linewidth": [lwidth_fid.data, lwidth_fid.header]}

    mvc = MVC_Distance(data_fid, data)
    mvc.distance_metric(verbose=True, xunit=u.pix**-1,
                        save_name=osjoin(fig_path, 'mvc_distmet.png'))
    print(mvc.distance)

    # With bounds
    mvc = MVC_Distance(data_fid, data, low_cut=0.02 / u.pix,
                       high_cut=0.1 / u.pix)
    mvc.distance_metric(verbose=True, xunit=u.pix**-1,
                        save_name=osjoin(fig_path, 'mvc_distmet_lims.png'))
    print(mvc.distance)

# PCA
if run_pca:

    from turbustat.statistics import PCA_Distance

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]
    cube_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc.fits"))[0]

    pca = PCA_Distance(cube_fid, cube, n_eigs=50, mean_sub=True)
    pca.distance_metric(verbose=True,
                        save_name=osjoin(fig_path, 'pca_distmet.png'))

    print(pca.distance)

# PDF
if run_pdf:

    from turbustat.statistics import PDF_Distance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[0]

    pdf = PDF_Distance(moment0_fid, moment0, min_val1=0.0, min_val2=0.0,
                       do_fit=True, normalization_type=None)
    pdf.distance_metric(verbose=True,
                        save_name=osjoin(fig_path, "pdf_distmet.png"))

    print(pdf.hellinger_distance)
    print(pdf.ks_distance)
    print(pdf.lognormal_distance)


# PSpec
if run_pspec:

    from turbustat.statistics import PSpec_Distance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[0]

    pspec = PSpec_Distance(moment0_fid, moment0,
                           low_cut=0.025 / u.pix, high_cut=0.1 / u.pix,)
    pspec.distance_metric(verbose=True, xunit=u.pix**-1,
                          save_name=osjoin(fig_path, "pspec_distmet.png"))

    print(pspec.distance)


# SCF
if run_scf:

    from turbustat.statistics import SCF_Distance

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]
    cube_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc.fits"))[0]

    scf = SCF_Distance(cube_fid, cube, size=11)
    scf.distance_metric(verbose=True,
                        save_name=osjoin(fig_path, "scf_distmet.png"))

    print(scf.distance)

# Moments
if run_moments:

    from turbustat.statistics import StatMoments_Distance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[0]

    moments = StatMoments_Distance(moment0_fid, moment0, radius=5 * u.pix,
                                   periodic1=True, periodic2=True)
    moments.distance_metric(verbose=True,
                            save_name=osjoin(fig_path, "statmoments_distmet.png"))


# Tsallis
# if run_tsallis:
#     from turbustat.statistics import Tsallis

#     moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]

#     tsallis = Tsallis(moment0).run(verbose=True,
#                                    save_name=osjoin(fig_path, 'design4_tsallis.png'))

#     # Tsallis parameters plot.
#     tsallis.plot_parameters(save_name=osjoin(fig_path, 'design4_tsallis_params.png'))

#     # With physical lags units
#     phys_lags = np.arange(0.025, 0.5, 0.05) * u.pc
#     tsallis = Tsallis(moment0, lags=phys_lags, distance=250 * u.pc)
#     tsallis.run(verbose=True,
#                 save_name=osjoin(fig_path, 'design4_tsallis_physlags.png'))

#     # Not periodic
#     tsallis_noper = Tsallis(moment0).run(verbose=True, periodic=False,
#                                          save_name=osjoin(fig_path, 'design4_tsallis_noper.png'))

#     # Change sigma clip
#     tsallis = Tsallis(moment0).run(verbose=True, sigma_clip=3,
#                                    save_name=osjoin(fig_path, 'design4_tsallis_sigclip.png'))

#     tsallis.plot_parameters(save_name=osjoin(fig_path, 'design4_tsallis_params_sigclip.png'))

# VCA
if run_vca:

    from turbustat.statistics import VCA_Distance

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]
    cube_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc.fits"))[0]

    vca = VCA_Distance(cube_fid, cube, low_cut=0.025 / u.pix,
                       high_cut=0.1 / u.pix)
    vca.distance_metric(verbose=True,
                        save_name=osjoin(fig_path, "vca_distmet.png"))

    print(vca.distance)

    vca = VCA_Distance(cube_fid, cube, low_cut=0.025 / u.pix,
                       high_cut=0.1 / u.pix, channel_width=400 * u.m / u.s)
    vca.distance_metric(verbose=True,
                        save_name=osjoin(fig_path, "vca_distmet_thickchan.png"))
    print(vca.distance)


# VCS
if run_vcs:

    from turbustat.statistics import VCS_Distance

    cube = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc.fits"))[0]
    cube_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc.fits"))[0]

    vcs = VCS_Distance(cube_fid, cube, fit_kwargs=dict(high_cut=0.17 / u.pix))
    vcs.distance_metric(verbose=True,
                        save_name=osjoin(fig_path, "vcs_distmet.png"))

    print(vcs.large_scale_distance, vcs.small_scale_distance, vcs.distance, vcs.break_distance)

# Wavelets
if run_wavelet:
    from turbustat.statistics import Wavelet_Distance

    moment0 = fits.open(osjoin(data_path, "Design4_flatrho_0021_00_radmc_moment0.fits"))[0]
    moment0_fid = fits.open(osjoin(data_path, "Fiducial0_flatrho_0021_00_radmc_moment0.fits"))[0]

    wavelet = Wavelet_Distance(moment0_fid, moment0, xlow=2 * u.pix,
                               xhigh=10 * u.pix)
    wavelet.distance_metric(verbose=True,
                            save_name=osjoin(fig_path, 'wavelet_distmet.png'))

    print(wavelet.distance)
