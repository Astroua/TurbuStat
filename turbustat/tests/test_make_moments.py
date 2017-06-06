# Licensed under an MIT open source license - see LICENSE

import numpy.testing as npt
import astropy.units as u

from ..data_reduction import Mask_and_Moments
from ._testing_data import dataset1, sc1


def test_loading():

    # Try loading the files.
    # Set the scale to the assumed value.
    test = Mask_and_Moments.from_fits(sc1, moments_prefix="dataset1",
                                      moments_path=".",
                                      scale=0.003031065017916262 * u.Unit(""))

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
