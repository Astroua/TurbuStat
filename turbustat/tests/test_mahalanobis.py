# # Licensed under an MIT open source license - see LICENSE
# from __future__ import print_function, absolute_import, division

# import pytest
# import warnings

# from ..statistics import Mahalanobis, Mahalanobis_Distance
# from ..statistics.stats_warnings import TurbuStatTestingWarning
# from ._testing_data import dataset1


# def test_Mahalanobis_raisewarning():
#     '''
#     Mahalanobis has not been completed yet. Ensure the warning is returned
#     when used.
#     '''

#     with warnings.catch_warnings(record=True) as w:
#         mahala = Mahalanobis(dataset1['cube'])

#     assert len(w) == 1
#     assert w[0].category == TurbuStatTestingWarning
#     assert str(w[0].message) == \
#         ("Mahalanobis is an untested statistic. Its use"
#          " is not yet recommended.")


# def test_Mahalanobis_Distance_raisewarning():
#     '''
#     Mahalanobis has not been completed yet. Ensure the warning is returned
#     when used.
#     '''

#     with warnings.catch_warnings(record=True) as w:
#         mahala = Mahalanobis_Distance(dataset1['cube'], dataset1['cube'])

#     # Warning is raised each time Mahalanobis is run (so twice)
#     assert len(w) == 3
#     assert w[0].category == TurbuStatTestingWarning
#     assert str(w[0].message) == \
#         ("Mahalanobis_Distance is an untested metric. Its use"
#          " is not yet recommended.")
