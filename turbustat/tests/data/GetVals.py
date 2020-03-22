# Licensed under an MIT open source license - see LICENSE


'''
Save the key results using the testing datasets.
'''

import pytest


@pytest.mark.skip(reason='Only generates unit values.')
def generate_unitvals():

    import numpy as np
    import astropy.units as u

    # The machine producing these values should have emcee installed!
    try:
        import emcee
    except ImportError:
        raise ImportError("Install emcee to generate unit test data.")

    from turbustat.tests._testing_data import dataset1, dataset2

    # Wavelet Transform

    from turbustat.statistics import Wavelet_Distance, Wavelet

    wavelet_distance = \
        Wavelet_Distance(dataset1["moment0"],
                         dataset2["moment0"]).distance_metric()

    wavelet_val = wavelet_distance.wt1.values
    wavelet_slope = wavelet_distance.wt1.slope

    # Wavelet with a break
    wave_break = Wavelet(dataset1['moment0']).run(xhigh=7 * u.pix, brk=5.5 * u.pix)

    wavelet_slope_wbrk = wave_break.slope
    wavelet_brk_wbrk = wave_break.brk.value

    # MVC

    from turbustat.statistics import MVC_Distance, MVC

    mvc_distance = MVC_Distance(dataset1, dataset2).distance_metric()

    mvc = MVC(dataset1["centroid"], dataset1["moment0"], dataset1["linewidth"])
    mvc.run()

    mvc_val = mvc.ps1D
    mvc_slope = mvc.slope
    mvc_slope2D = mvc.slope2D

    # Spatial Power Spectrum/ Bispectrum

    from turbustat.statistics import (PSpec_Distance, Bispectrum_Distance,
                                      Bispectrum, PowerSpectrum)

    pspec_distance = \
        PSpec_Distance(dataset1["moment0"],
                       dataset2["moment0"]).distance_metric()

    pspec = PowerSpectrum(dataset1['moment0'])
    pspec.run()

    pspec_val = pspec.ps1D
    pspec_slope = pspec.slope
    pspec_slope2D = pspec.slope2D

    bispec_distance = \
        Bispectrum_Distance(dataset1["moment0"],
                            dataset2["moment0"]).distance_metric()

    bispec_val = bispec_distance.bispec1.bicoherence

    azimuthal_slice = bispec_distance.bispec1.azimuthal_slice(16, 10,
                                                              value='bispectrum_logamp',
                                                              bin_width=5 * u.deg)
    bispec_azim_bins = azimuthal_slice[16][0]
    bispec_azim_vals = azimuthal_slice[16][1]
    bispec_azim_stds = azimuthal_slice[16][2]

    bispec_meansub = Bispectrum(dataset1['moment0'])
    bispec_meansub.run(mean_subtract=True)

    bispec_val_meansub = bispec_meansub.bicoherence

    # Genus

    from turbustat.statistics import GenusDistance, Genus

    smooth_scales = np.linspace(1.0, 0.1 * min(dataset1["moment0"][0].shape), 5)

    genus_distance = \
        GenusDistance(dataset1["moment0"],
                      dataset2["moment0"],
                      lowdens_percent=20,
                      genus_kwargs=dict(match_kernel=True)).distance_metric()

    # The distance method requires standardizing the data. Make a
    # separate version that isn't
    genus = Genus(dataset1['moment0'], smoothing_radii=smooth_scales)
    genus.run(match_kernel=True)

    genus_val = genus.genus_stats

    # Delta-Variance

    from turbustat.statistics import DeltaVariance_Distance, DeltaVariance

    delvar_distance = \
        DeltaVariance_Distance(dataset1["moment0"],
                               dataset2["moment0"],
                               weights1=dataset1["moment0_error"][0],
                               weights2=dataset2["moment0_error"][0],
                               delvar_kwargs=dict(xhigh=11 * u.pix))

    delvar_distance.distance_metric()

    delvar = DeltaVariance(dataset1["moment0"],
                           weights=dataset1['moment0_error'][0]).run(xhigh=11 * u.pix)

    delvar_val = delvar.delta_var
    delvar_slope = delvar.slope

    # Test with a break point
    delvar_wbrk = \
      DeltaVariance(dataset1["moment0"],
                    weights=dataset1['moment0_error'][0]).run(xhigh=11 * u.pix,
                                                              brk=6 * u.pix)

    delvar_slope_wbrk = delvar_wbrk.slope
    delvar_brk = delvar_wbrk.brk.value

    # Change boundary conditions

    delvar_fill = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1['moment0_error'][0]).run(xhigh=11 * u.pix,
                                                                boundary='fill',
                                                                nan_treatment='interpolate')

    delvar_fill_val = delvar_fill.delta_var
    delvar_fill_slope = delvar_fill.slope

    # VCA/VCS

    from turbustat.statistics import VCA_Distance, VCS_Distance, VCA

    vcs_distance = VCS_Distance(dataset1["cube"],
                                dataset2["cube"],
                                fit_kwargs=dict(high_cut=0.3 / u.pix,
                                                low_cut=3e-2 / u.pix))
    vcs_distance.distance_metric()

    vcs_val = vcs_distance.vcs1.ps1D
    vcs_slopes = vcs_distance.vcs1.slope

    vca_distance = VCA_Distance(dataset1["cube"],
                                dataset2["cube"]).distance_metric()

    vca = VCA(dataset1['cube'])
    vca.run()

    vca_val = vca.ps1D
    vca_slope = vca.slope
    vca_slope2D = vca.slope2D

    # Tsallis

    from turbustat.statistics import Tsallis

    tsallis_kwargs = {"sigma_clip": 5, "num_bins": 100}

    tsallis = Tsallis(dataset1['moment0'],
                      lags=[1, 2, 4, 8, 16] * u.pix)
    tsallis.run(periodic=True, **tsallis_kwargs)

    tsallis_val = tsallis.tsallis_params
    tsallis_stderrs = tsallis.tsallis_stderrs

    tsallis_noper = Tsallis(dataset1['moment0'],
                            lags=[1, 2, 4, 8, 16] * u.pix)
    tsallis_noper.run(periodic=False, num_bins=100)

    tsallis_val_noper = tsallis_noper.tsallis_params

    # High-order stats

    from turbustat.statistics import StatMoments_Distance, StatMoments

    moment_distance = \
        StatMoments_Distance(dataset1["moment0"],
                             dataset2["moment0"]).distance_metric()

    kurtosis_val = moment_distance.moments1.kurtosis_hist[1]
    skewness_val = moment_distance.moments1.skewness_hist[1]

    # Save a few from the non-distance version
    tester = StatMoments(dataset1["moment0"])
    tester.run()

    kurtosis_nondist_val = tester.kurtosis_hist[1]
    skewness_nondist_val = tester.skewness_hist[1]

    # Non-periodic
    tester = StatMoments(dataset1["moment0"])
    tester.run(periodic=False)

    kurtosis_nonper_val = tester.kurtosis_hist[1]
    skewness_nonper_val = tester.skewness_hist[1]


    # PCA

    from turbustat.statistics import PCA_Distance, PCA
    pca_distance = PCA_Distance(dataset1["cube"],
                                dataset2["cube"]).distance_metric()

    pca = PCA(dataset1["cube"], distance=250 * u.pc)
    pca.run(mean_sub=True, eigen_cut_method='proportion',
            min_eigval=0.75,
            spatial_method='contour',
            spectral_method='walk-down',
            fit_method='odr', brunt_beamcorrect=False,
            spectral_output_unit=u.m / u.s)

    pca_val = pca.eigvals
    pca_spectral_widths = pca.spectral_width().value
    pca_spatial_widths = pca.spatial_width().value

    pca_fit_vals = {"index": pca.index, "gamma": pca.gamma,
                    "intercept": pca.intercept().value,
                    "sonic_length": pca.sonic_length()[0].value}

    # Now get those values using mcmc
    pca.run(mean_sub=True, eigen_cut_method='proportion',
            min_eigval=0.75,
            spatial_method='contour',
            spectral_method='walk-down',
            fit_method='bayes', brunt_beamcorrect=False,
            spectral_output_unit=u.m / u.s)

    pca_fit_vals["index_bayes"] = pca.index
    pca_fit_vals["gamma_bayes"] = pca.gamma
    pca_fit_vals["intercept_bayes"] = pca.intercept().value
    pca_fit_vals["sonic_length_bayes"] = pca.sonic_length()[0].value

    # Record the number of eigenvalues kept by the auto method
    pca.run(mean_sub=True, n_eigs='auto', min_eigval=0.001,
            eigen_cut_method='value', decomp_only=True)

    pca_fit_vals["n_eigs_value"] = pca.n_eigs

    # Now w/ the proportion of variance cut
    pca.run(mean_sub=True, n_eigs='auto', min_eigval=0.99,
            eigen_cut_method='proportion', decomp_only=True)

    pca_fit_vals["n_eigs_proportion"] = pca.n_eigs

    # SCF

    from turbustat.statistics import SCF_Distance, SCF

    scf_distance = SCF_Distance(dataset1["cube"],
                                dataset2["cube"], size=11).distance_metric()

    scf = SCF(dataset1['cube'], size=11).run()

    scf_val = scf.scf_surface
    scf_spectrum = scf.scf_spectrum
    scf_slope = scf.slope
    scf_slope2D = scf.slope2D

    # Now run the SCF when the boundaries aren't continuous
    scf_distance_cut_bound = SCF_Distance(dataset1["cube"],
                                          dataset2["cube"], size=11,
                                          boundary='cut').distance_metric()
    scf_val_noncon_bound = scf_distance_cut_bound.scf1.scf_surface

    scf_fitlims = SCF(dataset1['cube'], size=11)
    scf_fitlims.run(boundary='continuous', xlow=1.5 * u.pix,
                    xhigh=4.5 * u.pix)

    scf_slope_wlimits = scf_fitlims.slope
    scf_slope_wlimits_2D = scf_fitlims.slope2D

    # Cramer Statistic

    from turbustat.statistics import Cramer_Distance

    cramer_distance = Cramer_Distance(dataset1["cube"],
                                      dataset2["cube"],
                                      noise_value1=0.1,
                                      noise_value2=0.1).distance_metric(normalize=False)

    cramer_val = cramer_distance.data_matrix1

    # Dendrograms

    from turbustat.statistics import Dendrogram_Distance, Dendrogram_Stats

    min_deltas = np.logspace(-1.5, 0.5, 40)

    dendro_distance = Dendrogram_Distance(dataset1["cube"],
                                          dataset2["cube"],
                                          min_deltas=min_deltas).distance_metric()

    dendrogram_val = dendro_distance.dendro1.numfeatures

    # With periodic boundaries
    dendro = Dendrogram_Stats(dataset1['cube'], min_deltas=min_deltas)
    dendro.run(periodic_bounds=True)

    dendrogram_periodic_val = dendro.numfeatures

    # PDF

    from turbustat.statistics import PDF_Distance

    pdf_distance = \
        PDF_Distance(dataset1["moment0"],
                     dataset2["moment0"],
                     min_val1=0.05,
                     min_val2=0.05,
                     weights1=dataset1["moment0_error"][0]**-2.,
                     weights2=dataset2["moment0_error"][0]**-2.,
                     do_fit=False,
                     normalization_type='standardize')

    pdf_distance.distance_metric()

    pdf_val = pdf_distance.PDF1.pdf
    pdf_ecdf = pdf_distance.PDF1.ecdf
    pdf_bins = pdf_distance.bins

    # Do a fitted version of the PDF pca
    pdf_fit_distance = \
        PDF_Distance(dataset1["moment0"],
                     dataset2["moment0"],
                     min_val1=0.05,
                     min_val2=0.05,
                     do_fit=True,
                     normalization_type=None)

    pdf_fit_distance.distance_metric()

    np.savez_compressed('checkVals',
                        wavelet_val=wavelet_val,
                        wavelet_slope=wavelet_slope,
                        wavelet_slope_wbrk=wavelet_slope_wbrk,
                        wavelet_brk_wbrk=wavelet_brk_wbrk,
                        mvc_val=mvc_val,
                        mvc_slope=mvc_slope,
                        mvc_slope2D=mvc_slope2D,
                        pspec_val=pspec_val,
                        pspec_slope=pspec_slope,
                        pspec_slope2D=pspec_slope2D,
                        bispec_val=bispec_val,
                        bispec_azim_bins=bispec_azim_bins,
                        bispec_azim_vals=bispec_azim_vals,
                        bispec_azim_stds=bispec_azim_stds,
                        bispec_val_meansub=bispec_val_meansub,
                        genus_val=genus_val,
                        delvar_val=delvar_val,
                        delvar_slope=delvar_slope,
                        delvar_slope_wbrk=delvar_slope_wbrk,
                        delvar_brk=delvar_brk,
                        delvar_fill_val=delvar_fill_val,
                        delvar_fill_slope=delvar_fill_slope,
                        vcs_val=vcs_val,
                        vcs_slopes=vcs_slopes,
                        vca_val=vca_val,
                        vca_slope=vca_slope,
                        vca_slope2D=vca_slope2D,
                        tsallis_val=tsallis_val,
                        tsallis_stderrs=tsallis_stderrs,
                        tsallis_val_noper=tsallis_val_noper,
                        kurtosis_val=kurtosis_val,
                        skewness_val=skewness_val,
                        kurtosis_nondist_val=kurtosis_nondist_val,
                        skewness_nondist_val=skewness_nondist_val,
                        kurtosis_nonper_val=kurtosis_nonper_val,
                        skewness_nonper_val=skewness_nonper_val,
                        pca_val=pca_val,
                        pca_fit_vals=pca_fit_vals,
                        pca_spectral_widths=pca_spectral_widths,
                        pca_spatial_widths=pca_spatial_widths,
                        scf_val=scf_val,
                        scf_slope_wlimits=scf_slope_wlimits,
                        scf_slope_wlimits_2D=scf_slope_wlimits_2D,
                        scf_val_noncon_bound=scf_val_noncon_bound,
                        scf_spectrum=scf_spectrum,
                        scf_slope=scf_slope,
                        scf_slope2D=scf_slope2D,
                        cramer_val=cramer_val,
                        dendrogram_val=dendrogram_val,
                        dendrogram_periodic_val=dendrogram_periodic_val,
                        pdf_val=pdf_val,
                        pdf_bins=pdf_bins,
                        pdf_ecdf=pdf_ecdf)

    np.savez_compressed('computed_distances', mvc_distance=mvc_distance.distance,
                        pca_distance=pca_distance.distance,
                        vca_distance=vca_distance.distance,
                        pspec_distance=pspec_distance.distance,
                        scf_distance=scf_distance.distance,
                        wavelet_distance=wavelet_distance.distance,
                        delvar_curve_distance=delvar_distance.curve_distance,
                        delvar_slope_distance=delvar_distance.slope_distance,
                        # tsallis_distance=tsallis_distance.distance,
                        kurtosis_distance=moment_distance.kurtosis_distance,
                        skewness_distance=moment_distance.skewness_distance,
                        cramer_distance=cramer_distance.distance,
                        genus_distance=genus_distance.distance,
                        vcs_distance=vcs_distance.distance,
                        bispec_mean_distance=bispec_distance.mean_distance,
                        bispec_surface_distance=bispec_distance.surface_distance,
                        dendrohist_distance=dendro_distance.histogram_distance,
                        dendronum_distance=dendro_distance.num_distance,
                        pdf_hellinger_distance=pdf_distance.hellinger_distance,
                        pdf_ks_distance=pdf_distance.ks_distance,
                        pdf_lognorm_distance=pdf_fit_distance.lognormal_distance)
                        # pdf_ad_distance=pdf_distance.ad_distance)
