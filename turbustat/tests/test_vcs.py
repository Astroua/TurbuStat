# Licensed under an MIT open source license - see LICENSE

'''
Test functions for VCS
'''

import pytest

import numpy as np
import numpy.testing as npt

from ..statistics import VCS, VCS_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_VCS_method():
    tester = VCS(dataset1["cube"]).run()

    npt.assert_allclose(tester.ps1D, computed_data['vcs_val'])

    npt.assert_allclose(tester.slope, computed_data['vcs_slopes_val'])


def test_VCS_distance():
    tester_dist = \
        VCS_Distance(dataset1["cube"], dataset2["cube"])
    tester_dist = tester_dist.distance_metric()

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['vcs_distance'])

# Add tests for: VCS changing the spectral width, pixel and spectral units,
