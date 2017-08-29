# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division


class TurbuStatTestingWarning(Warning):
    '''
    Turbustat.statistics warning for untested methods.
    '''


class TurbuStatMetricWarning(Warning):
    '''
    Turbustat.statistics warning for misusing a distance metric.
    '''
