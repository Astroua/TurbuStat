# Licensed under an MIT open source license - see LICENSE

import numpy as np
import numpy.testing as npt

from ..statistics.pdf import PDF, PDF_Distance

from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_PDF():
    test = PDF(dataset1["moment0"],
               normalization_type="standardize", min_val=0.05,
               weights=dataset1["moment0_error"][0]**-2.,
               bins=computed_data['pdf_bins'])
    test.run(verbose=False, do_fit=False)

    npt.assert_almost_equal(test.pdf / test.pdf.sum(),
                            computed_data["pdf_val"])
    npt.assert_almost_equal(test.ecdf, computed_data["pdf_ecdf"])

    npt.assert_equal(np.median(test.data),
                     test.find_at_percentile(50))

    npt.assert_equal(50.,
                     test.find_percentile(np.median(test.data)))


def test_PDF_distance():
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
    pass
