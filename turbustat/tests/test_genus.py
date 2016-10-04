# Licensed under an MIT open source license - see LICENSE

'''
Test functions for Genus
'''

import numpy as np
import numpy.testing as npt
import astropy.units as u

from ..statistics import GenusDistance, Genus
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_Genus_method():

    # The test values were normalized for generating the unit test data
    from ..statistics.stats_utils import standardize
    from copy import copy

    mom0 = copy(dataset1["moment0"])
    mom0[0] = standardize(mom0[0])

    tester = Genus(mom0, lowdens_percent=20)
    tester.run()

    assert np.allclose(tester.genus_stats,
                       computed_data['genus_val'])


def test_Genus_method_headerbeam():

    # The test values were normalized for generating the unit test data
    from ..statistics.stats_utils import standardize
    from copy import copy

    mom0 = copy(dataset1["moment0"])
    mom0[0] = standardize(mom0[0])
    mom0[1]["BMAJ"] = 1.0

    # Just ensuring these run without issue.

    tester = Genus(mom0)
    tester.run(use_beam=True)

    tester = Genus(mom0)
    tester.run(use_beam=True, beam_area=1.0 * u.sr)


def test_Genus_distance():
    tester_dist = \
        GenusDistance(dataset1["moment0"],
                      dataset2["moment0"])
    tester_dist.distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['genus_distance'])
