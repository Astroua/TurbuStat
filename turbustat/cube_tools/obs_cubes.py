
'''
Wrapper class for handling observational datasets
'''

import numpy as np

from spectral_cube import SpectralCube
# from signal_id import Mask

#  from cleaning_algs import *


class ObsCube(object):
    """docstring for ObsCube"""
    def __init__(self, cube, cleaning_alg=None):
        super(ObsCube, self).__init__()
        self.cube = SpectralCube.read(cube)

        self.cleaning_alg = cleaning_alg

    def apply_cleaning(self):

        return self

    def apply_mask(self):
        # self.cube = Mask(self.cube)

        return self
