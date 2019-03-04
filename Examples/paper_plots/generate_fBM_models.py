
'''
Generate optically-thin PPV cubes from fBM velocity and density fields.

* Velocity assumed to be isotropic
* Density field drawn from an fBM field, then the std. is added and values
  below 0 are set 0.
  (See Ossenkopf+2006;
   https://ui.adsabs.harvard.edu/#abs/2006A&A...452..223O/abstract)
* Thermal velocity set assuming 100 K emitters. See Esquivel+2003 and
  Chepurnov+2009 for discussions on 'shot noise' from a finite set of emitters
  along the line-of-sight.

'''

import os
from os.path import join as osjoin

import numpy as np
from astropy import units as u
from astropy import constants as cc
from astropy.utils.console import ProgressBar
from itertools import product
from astropy.utils import NumpyRNGContext
from scipy.stats import linregress

from turbustat.simulator import make_3dfield, make_ppv
from turbustat.simulator.threeD_pspec import threeD_pspec


out_dir = os.path.expanduser("~/MyRAID/Astrostat/TurbuStat_Paper/fBM_cubes")

# Generate a number of cubes with different velocity and density indices
vel_inds = np.array([3.3, 11 / 3., 4.])
dens_inds = np.array([2.5, 3., 11 / 3., 4.])

vel_amp = 10.  # km / s
dens_amp = 1.  # cm^-3

n_reps = 4

cube_sizes = [256, 512]

T = 100 * u.K
v_th = np.sqrt(cc.k_B * T / (1.4 * cc.m_p)).to(u.km / u.s)


# Draw random seeds for each repetition
with NumpyRNGContext(3458953087):

    rand_seeds_vel = np.random.randint(0, 2**31 - 1, size=n_reps)
    rand_seeds_dens = np.random.randint(0, 2**31 - 1, size=n_reps)

# Loop over the different params
bar = ProgressBar(len(vel_inds) * len(dens_inds) * n_reps * len(cube_sizes))

for cube_size in cube_sizes:

    # Real slopes will have small deviations from the assigned index
    vel_slopes = []
    dens_slopes = []

    # Define a common channel width with the steepest velocity index (1. -> -4)
    dv_eff = np.sqrt(vel_amp**2 * (2. / cube_size)**(vel_inds[-1] - 3.) +
                     2 * v_th.value**2) / 5.
    dv_eff *= u.km / u.s

    for dens, vel in product(dens_inds, vel_inds):

        for i, (rseed_dens, rseed_vel) in enumerate(zip(rand_seeds_dens,
                                                        rand_seeds_vel)):

            velocity = make_3dfield(cube_size, powerlaw=vel, amp=vel_amp,
                                    randomseed=rseed_vel) * u.km / u.s

            # Deal with negative density values.
            density = make_3dfield(cube_size, powerlaw=dens, amp=dens_amp,
                                   randomseed=rseed_dens) * u.cm**-3
            density += density.std()
            density[density.value < 0.] = 0. * u.cm**-3

            # Save the raw 3D fields

            filename = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}_size_{3}_3D_vel.npy"\
                .format(np.abs(dens), np.abs(vel), i, cube_size)

            np.save(osjoin(out_dir, filename), velocity.value)

            filename = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}_size_{3}_3D_dens.npy"\
                .format(np.abs(dens), np.abs(vel), i, cube_size)

            np.save(osjoin(out_dir, filename), density.value)

            # Also fit and record the best-fit field index
            vel_spec = threeD_pspec(velocity.value)
            vel_slope = linregress(np.log10(vel_spec[0][:-1]),
                                   np.log10(vel_spec[1][:-1]))
            vel_slopes.append(vel_slope.slope)

            dens_spec = threeD_pspec(density.value)
            dens_slope = linregress(np.log10(dens_spec[0][:-1]),
                                    np.log10(dens_spec[1][:-1]))
            dens_slopes.append(dens_slope.slope)

            cube_hdu = make_ppv(velocity, density, vel_disp=np.std(velocity),
                                T=T, threads=4, verbose=True,
                                chan_width=dv_eff / 2.,
                                v_min=-60 * u.km / u.s, v_max=60 * u.km / u.s,
                                max_chan=2000)

            filename = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}_size_{3}.fits"\
                .format(np.abs(dens), np.abs(vel), i, cube_size)

            cube_hdu.writeto(osjoin(out_dir, filename), overwrite=True)

            bar.update()


    filename = "fBM_3D_velocity_slopes_size_{0}.npy"\
        .format(cube_size)

    np.save(osjoin(out_dir, filename), np.array(vel_slopes))

    filename = "fBM_3D_density_slopes_size_{0}.npy"\
        .format(cube_size)

    np.save(osjoin(out_dir, filename), np.array(dens_slopes))
