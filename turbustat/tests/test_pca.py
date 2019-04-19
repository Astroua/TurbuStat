# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u
import astropy.constants as const
import os

try:
    import emcee
    EMCEE_INSTALLED = True
except ImportError:
    EMCEE_INSTALLED = False

from ..statistics import PCA, PCA_Distance
from ..statistics.pca.width_estimate import WidthEstimate1D, WidthEstimate2D
from ._testing_data import (dataset1, dataset2, computed_data,
                            computed_distances)
from .generate_test_images import generate_2D_array, generate_1D_array
from .testing_utilities import assert_between


def test_PCA_method():
    tester = PCA(dataset1["cube"], distance=250 * u.pc)
    tester.run(mean_sub=True, eigen_cut_method='proportion',
               min_eigval=0.75,
               spatial_method='contour',
               spectral_method='walk-down',
               fit_method='odr', brunt_beamcorrect=False,
               verbose=True, save_name='test.png')
    os.system("rm test.png")

    slice_used = slice(0, tester.n_eigs)
    npt.assert_allclose(tester.eigvals[slice_used],
                        computed_data['pca_val'][slice_used])

    npt.assert_allclose(tester.spatial_width().value,
                        computed_data['pca_spatial_widths'])
    npt.assert_allclose(tester.spectral_width(unit=u.pix).value,
                        computed_data['pca_spectral_widths'])

    fit_values = computed_data["pca_fit_vals"].reshape(-1)[0]
    assert_between(fit_values["index"], tester.index_error_range[0],
                   tester.index_error_range[1])
    assert_between(fit_values["gamma"], tester.gamma_error_range[0],
                   tester.gamma_error_range[1])
    assert_between(fit_values["intercept"],
                   tester.intercept_error_range(unit=u.pix)[0].value,
                   tester.intercept_error_range(unit=u.pix)[1].value)
    assert_between(fit_values["sonic_length"],
                   tester.sonic_length(use_gamma=True)[1][0].value,
                   tester.sonic_length(use_gamma=True)[1][1].value)

    # Try setting fit limits to the max and min and ensure we get the same
    # values out
    tester.fit_plaw(xlow=np.nanmin(tester.spatial_width()),
                    xhigh=np.nanmax(tester.spatial_width()))

    assert_between(fit_values["index"], tester.index_error_range[0],
                   tester.index_error_range[1])
    assert_between(fit_values["gamma"], tester.gamma_error_range[0],
                   tester.gamma_error_range[1])
    assert_between(fit_values["intercept"],
                   tester.intercept_error_range(unit=u.pix)[0].value,
                   tester.intercept_error_range(unit=u.pix)[1].value)
    assert_between(fit_values["sonic_length"],
                   tester.sonic_length(use_gamma=True)[1][0].value,
                   tester.sonic_length(use_gamma=True)[1][1].value)

    # Test loading and saving
    tester.save_results("pca_output.pkl", keep_data=False)

    saved_tester = PCA.load_results("pca_output.pkl")

    # Remove the file
    os.remove("pca_output.pkl")

    npt.assert_allclose(saved_tester.eigvals[slice_used],
                        computed_data['pca_val'][slice_used])

    npt.assert_allclose(saved_tester.spatial_width().value,
                        computed_data['pca_spatial_widths'])
    npt.assert_allclose(saved_tester.spectral_width(unit=u.pix).value,
                        computed_data['pca_spectral_widths'])

    fit_values = computed_data["pca_fit_vals"].reshape(-1)[0]
    assert_between(fit_values["index"], saved_tester.index_error_range[0],
                   saved_tester.index_error_range[1])
    assert_between(fit_values["gamma"], saved_tester.gamma_error_range[0],
                   saved_tester.gamma_error_range[1])
    assert_between(fit_values["intercept"],
                   saved_tester.intercept_error_range(unit=u.pix)[0].value,
                   saved_tester.intercept_error_range(unit=u.pix)[1].value)
    assert_between(fit_values["sonic_length"],
                   saved_tester.sonic_length(use_gamma=True)[1][0].value,
                   saved_tester.sonic_length(use_gamma=True)[1][1].value)


@pytest.mark.skipif("not EMCEE_INSTALLED")
def test_PCA_method_w_bayes():
    tester = PCA(dataset1["cube"])
    tester.run(mean_sub=True, eigen_cut_method='proportion',
               min_eigval=0.75,
               spatial_method='contour',
               spectral_method='walk-down',
               fit_method='bayes', brunt_beamcorrect=False,
               spectral_output_unit=u.m / u.s)
    slice_used = slice(0, tester.n_eigs)
    npt.assert_allclose(tester.eigvals[slice_used],
                        computed_data['pca_val'][slice_used])

    npt.assert_allclose(tester.spatial_width().value,
                        computed_data['pca_spatial_widths'])
    npt.assert_allclose(tester.spectral_width(unit=u.pix).value,
                        computed_data['pca_spectral_widths'])

    fit_values = computed_data["pca_fit_vals"].reshape(-1)[0]
    assert_between(fit_values["index_bayes"], tester.index_error_range[0],
                   tester.index_error_range[1])
    assert_between(fit_values["gamma_bayes"], tester.gamma_error_range[0],
                   tester.gamma_error_range[1])
    assert_between(fit_values["intercept_bayes"],
                   tester.intercept_error_range(unit=u.pix)[0].value,
                   tester.intercept_error_range(unit=u.pix)[1].value)
    assert_between(fit_values["sonic_length_bayes"],
                   tester.sonic_length(use_gamma=True)[1][0].value,
                   tester.sonic_length(use_gamma=True)[1][1].value)


@pytest.mark.parametrize("method", ['odr', 'bayes'])
@pytest.mark.skipif("not EMCEE_INSTALLED")
def test_PCA_fitting(method):

    tester = PCA(dataset1["cube"])

    index = 2.
    intercept = 1.
    err = 0.02

    tester._spectral_width = (intercept * np.arange(10)**index +
                              err * np.random.random(10)) * u.pix
    tester._spectral_width_error = np.array([err] * 10) * u.pix
    tester._spatial_width = (np.arange(10) + err * np.random.random(10)) * \
        u.pix
    tester._spatial_width_error = np.array([err] * 10) * u.pix

    tester.fit_plaw(fit_method=method)

    npt.assert_allclose(tester.index, index, atol=0.05)
    npt.assert_allclose(tester.intercept(unit=u.pix).value, 1, atol=0.05)
    npt.assert_allclose(tester.gamma, 1.52 * index - 0.19, atol=0.05)

    # Check the sonic length
    T_k = 10 * u.K
    mu = 1.36
    c_s = np.sqrt(const.k_B.decompose() * T_k / (mu * const.m_p))
    # Convert into number of spectral channel widths
    c_s = c_s.to(u.m / u.s).value / np.abs(tester.header['CDELT3'])
    l_s = np.power(c_s / 1., 1. / index)

    npt.assert_allclose(l_s,
                        tester.sonic_length(use_gamma=False)[0].value,
                        atol=0.05)


@pytest.mark.parametrize(("method", "min_eigval"),
                         [("proportion", 0.99), ("value", 0.001)])
def test_PCA_auto_n_eigs(method, min_eigval):
    tester = PCA(dataset1["cube"])
    tester.run(mean_sub=True, n_eigs='auto', min_eigval=min_eigval,
               eigen_cut_method=method, decomp_only=True)

    fit_values = computed_data["pca_fit_vals"].reshape(-1)[0]
    assert tester.n_eigs == fit_values["n_eigs_" + method]


def test_PCA_distance():
    tester_dist = \
        PCA_Distance(dataset1["cube"],
                     dataset2["cube"])
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['pca_distance'])

    # With PCA classes as inputs
    tester_dist2 = \
        PCA_Distance(tester_dist.pca1,
                     tester_dist.pca2)
    tester_dist2.distance_metric(verbose=False)

    npt.assert_almost_equal(tester_dist2.distance,
                            computed_distances['pca_distance'])

    # With fresh PCA classes
    tester = PCA(dataset1["cube"])
    tester2 = PCA(dataset2["cube"])

    tester_dist3 = \
        PCA_Distance(tester,
                     tester2)
    tester_dist3.distance_metric(verbose=False)

    npt.assert_almost_equal(tester_dist3.distance,
                            computed_distances['pca_distance'])


@pytest.mark.parametrize(('method'), ('fit', 'contour', 'interpolate',
                                      'xinterpolate'))
def test_spatial_width_methods(method):
    '''
    Generate a 2D gaussian and test whether each method returns the expected
    size.

    Note that, as defined by Heyer & Brunt, the shape will be sigma / sqrt(2),
    NOT just the Gaussian width equivalent!
    '''

    model_gauss = generate_2D_array(x_std=10, y_std=10)

    model_gauss += np.random.normal(loc=0.0, scale=0.001,
                                    size=model_gauss.shape)

    model_gauss = model_gauss[np.newaxis, :]

    widths, errors = WidthEstimate2D(model_gauss, method=method,
                                     brunt_beamcorrect=False)

    # NOTE: previous versions were testing against 10 / 2**0.5
    # THIS WAS WRONG!
    npt.assert_allclose(widths[0], 10.0 * np.sqrt(2), rtol=0.02)
    # npt.assert_approx_equal(widths[0], 10.0 / np.sqrt(2), significant=3)
    # I get 0.000449 for the error, but we're in a noiseless case so just
    # ensure that is very small.
    assert errors[0] < 0.2


def test_spatial_with_beam():
    '''
    Test running the spatial width find with beam corrections enabled.
    '''

    conv_scale = np.sqrt(10**2 + (4 / 2.35)**2)

    model_gauss = generate_2D_array(x_std=conv_scale, y_std=conv_scale)

    model_gauss = model_gauss[np.newaxis, :]

    widths, errors = WidthEstimate2D(model_gauss, method='contour',
                                     brunt_beamcorrect=True,
                                     beam_fwhm=2.0 * u.deg,
                                     spatial_cdelt=0.5 * u.deg)

    # Using value based on run with given settings.
    npt.assert_approx_equal(widths[0], 13.9229, significant=4)


@pytest.mark.parametrize(('method'), ('fit', 'interpolate', 'walk-down'))
def test_spectral_width_methods(method):
    '''
    Generate a 1D gaussian and test whether each method returns the expected
    size.
    '''

    model_gauss = generate_1D_array(std=10, mean=100.)
    # Apply same shift as if taking FFT
    shiftx = np.fft.fftshift(model_gauss)[:, np.newaxis]

    widths, errors = WidthEstimate1D(shiftx, method=method)

    # Expected 1/e width is sqrt(2) * sigma
    # Error is at most 1/2 a spectral channel, or just 0.5 in this case
    npt.assert_allclose(widths[0], 10.0 * np.sqrt(2), rtol=0.01)


@pytest.mark.xfail(raises=Warning)
def test_PCA_velocity_axis():
    '''
    PCA requires a velocity spectral axis.
    '''

    new_hdr = dataset1["cube"][1].copy()

    new_hdr["CTYPE3"] = "FREQ    "
    new_hdr["CUNIT3"] = "Hz      "

    PCA([dataset1["cube"][0], new_hdr])
