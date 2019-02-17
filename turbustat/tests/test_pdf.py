# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import numpy.testing as npt
import os

from ..statistics.pdf import PDF, PDF_Distance

from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_PDF():
    test = PDF(dataset1["moment0"],
               normalization_type="standardize", min_val=0.05,
               weights=dataset1["moment0_error"][0]**-2.,
               bins=computed_data['pdf_bins'])
    test.run(verbose=True, do_fit=False, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(test.pdf,
                            computed_data["pdf_val"])
    npt.assert_almost_equal(test.ecdf, computed_data["pdf_ecdf"])

    npt.assert_equal(np.median(test.data),
                     test.find_at_percentile(50))

    npt.assert_equal(50.,
                     test.find_percentile(np.median(test.data)))

    # Test loading and saving
    test.save_results("pdf_output.pkl", keep_data=True)

    saved_test = PDF.load_results("pdf_output.pkl")

    # Remove the file
    os.remove("pdf_output.pkl")

    npt.assert_almost_equal(saved_test.pdf,
                            computed_data["pdf_val"])
    npt.assert_almost_equal(saved_test.ecdf, computed_data["pdf_ecdf"])

    npt.assert_equal(np.median(saved_test.data),
                     saved_test.find_at_percentile(50))

    npt.assert_equal(50.,
                     saved_test.find_percentile(np.median(saved_test.data)))


def test_PDF_fitting():
    '''
    Test distribution fitting for PDFs

    By default, we use the lognormal distribution, and only test it here.
    '''

    from scipy.stats import lognorm
    from numpy.random import seed

    seed(13493099)

    data1 = lognorm.rvs(0.4, loc=0.0, scale=1.0, size=50000)

    test = PDF(data1).run()

    npt.assert_almost_equal(0.40, test.model_params[0], decimal=2)
    npt.assert_almost_equal(1.0, test.model_params[1], decimal=1)


def test_PDF_distance():
    '''
    Test the non-parametric distances
    '''
    test_dist = \
        PDF_Distance(dataset1["moment0"],
                     dataset2["moment0"],
                     min_val1=0.05,
                     min_val2=0.05,
                     weights1=dataset1["moment0_error"][0]**-2.,
                     weights2=dataset2["moment0_error"][0]**-2.,
                     do_fit=False,
                     normalization_type='standardize')
    test_dist.distance_metric()

    npt.assert_almost_equal(test_dist.hellinger_distance,
                            computed_distances['pdf_hellinger_distance'])

    npt.assert_almost_equal(test_dist.ks_distance,
                            computed_distances['pdf_ks_distance'])

    # npt.assert_almost_equal(self.test_dist.ad_distance,
    #                         computed_distances['pdf_ad_distance'])


def test_PDF_lognormal_distance():
    '''
    Test the lognormal width based distance measure.
    '''

    from scipy.stats import lognorm
    from numpy.random import seed

    seed(13493099)

    data1 = lognorm.rvs(0.4, loc=0.0, scale=1.0, size=5000)
    data2 = lognorm.rvs(0.5, loc=0.0, scale=1.0, size=5000)

    test_dist = \
        PDF_Distance(data1,
                     data2,
                     do_fit=True,
                     normalization_type='normalize_by_mean')
    test_dist.distance_metric()

    # Based on the samples, these are the expected stderrs.
    actual_dist = (0.5 - 0.4) / np.sqrt(0.004**2 + 0.005**2)

    # The distance value can scatter by a couple based on small variations.
    # With the seed set, this should always be true.
    assert np.abs(test_dist.lognormal_distance - actual_dist) < 2.


def test_PDF_lognorm_distance():
    '''
    Test the non-parametric distances
    '''
    test_dist = \
        PDF_Distance(dataset1["moment0"],
                     dataset2["moment0"],
                     min_val1=0.05,
                     min_val2=0.05,
                     do_fit=True,
                     normalization_type=None)
    test_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(test_dist.lognormal_distance,
                            computed_distances['pdf_lognorm_distance'],
                            decimal=4)
