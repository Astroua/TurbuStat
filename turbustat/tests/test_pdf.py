# Licensed under an MIT open source license - see LICENSE

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics.pdf import PDF, PDF_Distance

from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testPDF(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_PDF(self):
        self.test = PDF(self.dataset1["moment0"],
                        use_standardized=True, min_val=0.05,
                        weights=self.dataset1["moment0_error"][0]**-2.,
                        bins=computed_data['pdf_bins'])
        self.test.run(verbose=False)

        npt.assert_almost_equal(self.test.pdf, computed_data["pdf_val"])
        npt.assert_almost_equal(self.test.ecdf, computed_data["pdf_ecdf"])

        npt.assert_equal(np.median(self.test.data),
                         self.test.find_at_percentile(50))

        npt.assert_equal(50.,
                         self.test.find_percentile(np.median(self.test.data)))

    def test_PDF_distance(self):
        self.test_dist = \
            PDF_Distance(self.dataset1["moment0"],
                         self.dataset2["moment0"],
                         min_val1=0.05,
                         min_val2=0.05,
                         weights1=self.dataset1["moment0_error"][0]**-2.,
                         weights2=self.dataset2["moment0_error"][0]**-2.)
        self.test_dist.distance_metric()

        npt.assert_almost_equal(self.test_dist.hellinger_distance,
                                computed_distances['pdf_hellinger_distance'])

        npt.assert_almost_equal(self.test_dist.ks_distance,
                                computed_distances['pdf_ks_distance'])

        # npt.assert_almost_equal(self.test_dist.ad_distance,
        #                         computed_distances['pdf_ad_distance'])
