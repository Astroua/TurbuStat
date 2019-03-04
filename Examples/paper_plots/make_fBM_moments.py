
'''
Make moment maps from the fBM cubes.

Only generating from the repeated cubes with density and velocity indices
of -4.
'''


from spectral_cube import SpectralCube
from astropy.table import Table, Column
import numpy as np
import astropy.units as u
from astropy import constants as cc
import os
import matplotlib.pyplot as plt
from itertools import product

from turbustat.moments import Moments

# data_path = "/Volumes/Travel_Data/Turbulence/fBM_cubes/"
data_path = os.path.expanduser("~/MyRAID/Astrostat/TurbuStat_Paper/fBM_cubes/")

vel_inds = np.array([3.3, 11 / 3., 4.])
dens_inds = np.array([2.5, 3., 11 / 3., 4.])
reps = range(5)
cube_size = 256


spec_ext = []

# for dens, vel, rep in product(dens_inds, vel_inds, reps):
# for dens, vel in product(dens_inds[::-1], vel_inds[::-1]):
for rep in reps:

    dens = 4.
    vel = 4.

    name = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}"\
        .format(np.abs(dens), np.abs(vel), rep)

    filename = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}_size_{3}.fits"\
        .format(np.abs(dens), np.abs(vel), rep, cube_size)

    cube = SpectralCube.read(os.path.join(data_path, filename))

    moments = Moments(cube)
    moments.make_moments()
    moments.make_moment_errors()

