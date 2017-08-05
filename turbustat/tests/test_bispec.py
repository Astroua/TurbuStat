# Licensed under an MIT open source license - see LICENSE

import pytest

import numpy as np
import numpy.testing as npt

from ..statistics import BiSpectrum, BiSpectrum_Distance
from ._testing_data import dataset1,\
    dataset2, computed_data, computed_distances


def test_Bispec_method():
    tester = BiSpectrum(dataset1["moment0"])
    tester.run()
    assert np.allclose(tester.bicoherence,
                       computed_data['bispec_val'])


def test_Bispec_method_meansub():
    tester = BiSpectrum(dataset1["moment0"])
    tester.run(mean_subtract=True)
    assert np.allclose(tester.bicoherence,
                       computed_data['bispec_val_meansub'])

def test_Bispec_distance():
    tester_dist = \
        BiSpectrum_Distance(dataset1["moment0"],
                            dataset2["moment0"])
    tester_dist.distance_metric()

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['bispec_distance'])
