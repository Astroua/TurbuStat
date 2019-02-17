# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u
import os

from ..statistics import Tsallis  # , Tsallis_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_Tsallis():
    tester = Tsallis(dataset1["moment0"],
                     lags=[1, 2, 4, 8, 16] * u.pix)
    tester.run(num_bins=100, periodic=True, verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_allclose(tester.tsallis_params,
                        computed_data['tsallis_val'], atol=0.01)

    # Ensure that the output table matched the right inputs by checking a
    # couple columns
    npt.assert_allclose(tester.tsallis_table['logA'],
                        computed_data['tsallis_val'][:, 0])
    npt.assert_allclose(tester.tsallis_table['logA_stderr'],
                        computed_data['tsallis_stderrs'][:, 0])

    # Test loading and saving
    tester.save_results("tsallis_output.pkl", keep_data=False)

    saved_tester = Tsallis.load_results("tsallis_output.pkl")

    # Remove the file
    os.remove("tsallis_output.pkl")

    npt.assert_allclose(saved_tester.tsallis_params,
                        computed_data['tsallis_val'], atol=0.01)

    # Ensure that the output table matched the right inputs by checking a
    # couple columns
    npt.assert_allclose(saved_tester.tsallis_table['logA'],
                        computed_data['tsallis_val'][:, 0])
    npt.assert_allclose(saved_tester.tsallis_table['logA_stderr'],
                        computed_data['tsallis_stderrs'][:, 0])


def test_Tsallis_noper():
    tester = Tsallis(dataset1["moment0"],
                     lags=[1, 2, 4, 8, 16] * u.pix)
    tester.run(num_bins=100, periodic=False)
    npt.assert_allclose(tester.tsallis_params,
                        computed_data['tsallis_val_noper'], atol=0.01)

    # Ensure that the output table matched the right inputs by checking a
    # couple columns
    npt.assert_allclose(tester.tsallis_table['logA'],
                        computed_data['tsallis_val_noper'][:, 0])


def test_Tsallis_lagunits():

    pix_lags = [1, 2, 4, 8, 16] * u.pix

    ang_conv = np.abs(dataset1['moment0'][1]["CDELT2"]) * u.deg
    ang_lags = pix_lags.value * ang_conv

    distance = 250 * u.pc
    phys_conv = ang_conv.to(u.rad).value * distance
    phys_lags = pix_lags.value * phys_conv

    tester = Tsallis(dataset1["moment0"],
                     lags=ang_lags)
    tester.run(num_bins=100, periodic=True)
    npt.assert_allclose(tester.tsallis_params,
                        computed_data['tsallis_val'], atol=0.01)

    tester = Tsallis(dataset1["moment0"],
                     lags=phys_lags, distance=distance)
    tester.run(num_bins=100, periodic=True)
    npt.assert_allclose(tester.tsallis_params,
                        computed_data['tsallis_val'], atol=0.01)


# def test_Tsallis_distance():
#     kwarg_dict = dict(num_bins=100, periodic=True)
#     tester_dist = \
#         Tsallis_Distance(dataset1["moment0"],
#                          dataset2["moment0"],
#                          lags=[1, 2, 4, 8, 16] * u.pix,
#                          tsallis1_kwargs=kwarg_dict,
#                          tsallis2_kwargs=kwarg_dict).distance_metric()
#     npt.assert_almost_equal(tester_dist.distance,
#                             computed_distances['tsallis_distance'],
#                             decimal=4)
