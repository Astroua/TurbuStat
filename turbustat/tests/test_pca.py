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
                            generate_1D_array)


class testPCA(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_PCA_method(self):
        self.tester = PCA(dataset1["cube"], n_eigs=50)
        self.tester.run(mean_sub=True, spatial_method='contour',
                        spectral_method='walk-down',
                        fit_method='odr', beam_fwhm=0.01 * u.arcsec)
        npt.assert_allclose(self.tester.eigvals, computed_data['pca_val'])

        fit_values = computed_data["pca_fit_vals"].reshape(-1)[0]
        npt.assert_equal(self.tester.index, fit_values["index"])
        npt.assert_equal(self.tester.gamma, fit_values["gamma"])
        npt.assert_equal(self.tester.intercept, fit_values["intercept"])
        npt.assert_equal(self.tester.sonic_length()[0],
                         fit_values["sonic_length"])

    @pytest.mark.skipif("not EMCEE_INSTALLED")
    def test_PCA_method_w_bayes(self):
        self.tester = PCA(dataset1["cube"], n_eigs=50)
        self.tester.run(mean_sub=True, spatial_method='contour',
                        spectral_method='walk-down',
                        fit_method='bayes', beam_fwhm=0.01 * u.arcsec)
        npt.assert_allclose(self.tester.eigvals, computed_data['pca_val'])

        fit_values = computed_data["pca_fit_vals"].reshape(-1)[0]
        npt.assert_allclose(self.tester.index,
                            fit_values["index_bayes"],
                            atol=0.01)
        npt.assert_allclose(self.tester.gamma,
                            fit_values["gamma_bayes"],
                            atol=0.01)
        npt.assert_allclose(self.tester.intercept.value,
                            fit_values["intercept_bayes"].value,
                            atol=0.5)
        npt.assert_allclose(self.tester.sonic_length()[0].value,
                            fit_values["sonic_length_bayes"].value, atol=0.5)

    def test_PCA_distance(self):
        self.tester_dist = \
            PCA_Distance(dataset1["cube"],
                         dataset2["cube"]).distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
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

    model_gauss = model_gauss[np.newaxis, :]

    widths, errors = WidthEstimate2D(model_gauss, method=method,
                                     brunt_beamcorrect=False)

    npt.assert_approx_equal(widths[0], 10.0 / np.sqrt(2), significant=3)
    # I get 0.000449 for the error, but we're in a noiseless case so just
    # ensure that is very small.
    assert errors[0] < 0.1


def test_spatial_with_beam():
    '''
    Test running the spatial width find with beam corrections enabled.
    '''
    import astropy.units as u

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
