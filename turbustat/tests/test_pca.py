# Licensed under an MIT open source license - see LICENSE

'''
Test functions for PCA
'''

from unittest import TestCase
import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u


try:
    import emcee
    EMCEE_INSTALLED = True
except ImportError:
    EMCEE_INSTALLED = False

from ..statistics import PCA, PCA_Distance
from ..statistics.pca.width_estimate import WidthEstimate1D, WidthEstimate2D
from ._testing_data import (dataset1, dataset2, computed_data,
                            computed_distances, generate_2D_array,
                            generate_1D_array, assert_between)


def test_PCA_method():
    tester = PCA(dataset1["cube"])
    tester.run(mean_sub=True, n_eigs=50,
               spatial_method='contour',
               spectral_method='walk-down',
               fit_method='odr', brunt_beamcorrect=False)
    slice_used = slice(0, tester.n_eigs)
    npt.assert_allclose(tester.eigvals[slice_used],
                        computed_data['pca_val'][slice_used])

    fit_values = computed_data["pca_fit_vals"].reshape(-1)[0]
    assert_between(fit_values["index"], tester.index_error_range[0],
                   tester.index_error_range[1])
    assert_between(fit_values["gamma"], tester.gamma_error_range[0],
                   tester.gamma_error_range[1])
    assert_between(fit_values["intercept"].value,
                   tester.intercept_error_range[0].value,
                   tester.intercept_error_range[1].value)
    assert_between(fit_values["sonic_length"].value,
                   tester.sonic_length()[1][0].value,
                   tester.sonic_length()[1][1].value)


@pytest.mark.skipif("not EMCEE_INSTALLED")
def test_PCA_method_w_bayes():
    tester = PCA(dataset1["cube"])
    tester.run(mean_sub=True, n_eigs=50,
               spatial_method='contour',
               spectral_method='walk-down',
               fit_method='bayes', brunt_beamcorrect=False)
    slice_used = slice(0, tester.n_eigs)
    npt.assert_allclose(tester.eigvals[slice_used],
                        computed_data['pca_val'][slice_used])

    fit_values = computed_data["pca_fit_vals"].reshape(-1)[0]
    assert_between(fit_values["index_bayes"], tester.index_error_range[0],
                   tester.index_error_range[1])
    assert_between(fit_values["gamma_bayes"], tester.gamma_error_range[0],
                   tester.gamma_error_range[1])
    assert_between(fit_values["intercept_bayes"].value,
                   tester.intercept_error_range[0].value,
                   tester.intercept_error_range[1].value)
    assert_between(fit_values["sonic_length_bayes"].value,
                   tester.sonic_length()[1][0].value,
                   tester.sonic_length()[1][1].value)


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
                     dataset2["cube"]).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
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

    npt.assert_allclose(widths[0], 10.0 / np.sqrt(2), atol=0.02)
    # npt.assert_approx_equal(widths[0], 10.0 / np.sqrt(2), significant=3)
    # I get 0.000449 for the error, but we're in a noiseless case so just
    # ensure that is very small.
    assert errors[0] < 0.1


def test_spatial_with_beam():
    '''
    Test running the spatial width find with beam corrections enabled.
    '''

    model_gauss = generate_2D_array(x_std=10, y_std=10)

    model_gauss = model_gauss[np.newaxis, :]

    widths, errors = WidthEstimate2D(model_gauss, method='contour',
                                     brunt_beamcorrect=False,
                                     beam_fwhm=2.0 * u.deg,
                                     spatial_cdelt=0.5 * u.deg)

    # Using value based on run with given settings.
    npt.assert_approx_equal(widths[0], 7.071, significant=4)


@pytest.mark.parametrize(('method'), ('fit', 'interpolate', 'walk-down'))
def test_spectral_width_methods(method):
    '''
    Generate a 1D gaussian and test whether each method returns the expected
    size.
    '''

    model_gauss = generate_1D_array(std=10, mean=100.)

    fftx = np.fft.fft(model_gauss)
    fftxs = np.conjugate(fftx)
    acor = np.fft.ifft((fftx - fftx.mean()) * (fftxs - fftxs.mean())).real

    # Should always be normalized such that the max is 1.
    acor = acor[:, np.newaxis] / acor.max()

    widths, errors = WidthEstimate1D(acor, method=method)

    # Error is at most 1/2 a spectral channel, or just 0.5 in this case
    npt.assert_allclose(widths[0], 10.0, atol=errors[0])


@pytest.mark.xfail(raises=Warning)
def test_PCA_velocity_axis():
    '''
    PCA requires a velocity spectral axis.
    '''

    new_hdr = dataset1["cube"][1].copy()

    new_hdr["CTYPE3"] = "FREQ    "
    new_hdr["CUNIT3"] = "Hz      "

    PCA([dataset1["cube"][0], new_hdr])
