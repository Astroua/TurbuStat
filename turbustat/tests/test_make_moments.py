# Licensed under an MIT open source license - see LICENSE

from unittest import TestCase

import numpy.testing as npt


from ..data_reduction import Mask_and_Moments
from ._testing_data import dataset1, sc1


class test_Mask_and_Moments(TestCase):
    """docstring for test_Mask_and_Moments"""

    def test_loading(self):

        # Try loading the files.

        test = Mask_and_Moments.from_fits(sc1, moments_prefix="dataset1",
                                          moments_path=".")

        npt.assert_allclose(test.moment0, dataset1["moment0"][0])
        npt.assert_allclose(test.moment1, dataset1["centroid"][0])
        npt.assert_allclose(test.linewidth, dataset1["linewidth"][0])
        npt.assert_allclose(test.intint,
                            dataset1["integrated_intensity"][0])

        npt.assert_allclose(test.moment0_err, dataset1["moment0_error"][0])
        npt.assert_allclose(test.moment1_err, dataset1["centroid_error"][0])
        npt.assert_allclose(test.linewidth_err, dataset1["linewidth_error"][0])
        npt.assert_allclose(test.intint_err,
                            dataset1["integrated_intensity_error"][0])
