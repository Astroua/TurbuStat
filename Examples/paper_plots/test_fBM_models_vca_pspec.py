
'''
Run different statistics on the fBM cubes
'''

from spectral_cube import SpectralCube
from astropy.table import Table, Column
import numpy as np
import astropy.units as u
from astropy import constants as cc
import os
import matplotlib.pyplot as plt
from itertools import product

from turbustat.statistics import PowerSpectrum, VCA

# data_path = "/Volumes/Travel_Data/Turbulence/fBM_cubes/"
data_path = os.path.expanduser("~/MyRAID/Astrostat/TurbuStat_Paper/fBM_cubes/")

# vel_inds = np.array([3.3, 11 / 3., 4.])
# dens_inds = np.array([2.5, 3., 11 / 3., 4.])

vel_inds = np.array([4.])
dens_inds = np.array([11 / 3., 4.])

reps = range(4)
cube_size = 256  # 128  # 256

# Will want to vary the slice size.
pixs = [1, 2, 4, 8, 16, 32, 64, 128]

outputs = {"v_thick_exp": [],
           "thin_exp": [],
           "VCA": [],
           "VCA_2D": [],
           "pspec": [],
           "pspec_2D": []}


def esquivel_max_k(sigma_L, chan_width, v_th, m):

    return (sigma_L**(-2) * (chan_width**2 + 2 * v_th**2))**(-1. / m)


cubes = []

for dens, vel, rep in product(dens_inds, vel_inds, reps):

    # Expected slopes
    vca_exp = 0.5 * (9 - vel) if dens >= 3 else 0.5 * (2 * dens - vel + 3)
    vca_thick_exp = dens if dens < 3 else 3 - 0.5 * (3 - vel)
    pspec_exp = dens

    name = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}"\
        .format(np.abs(dens), np.abs(vel), rep)

    filename = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}_size_{3}.fits"\
        .format(np.abs(dens), np.abs(vel), rep, cube_size)

    cube = SpectralCube.read(os.path.join(data_path, filename))
    cube.allow_huge_operations = True

    # cube = cube.spectral_slab(-30 * u.km / u.s, 30 * u.km / u.s)

    # cubes.append(cube)

    mom0 = cube.moment0()

    vca_slopes = []
    vca_slopes_2D = []

    vel_size = []

    # Estimate the upper limit freq. by Eq 14 in Esquivel+2003
    chan_width = np.diff(cube.spectral_axis[:2])[0].value
    T = 100 * u.K
    v_th = np.sqrt(cc.k_B * T / (1.4 * cc.m_p)).to(u.km / u.s)

    chan_width = np.diff(cube.spectral_axis[:2])[0].value
    high_cut = esquivel_max_k(10., chan_width, v_th.value, vel - 3.) / \
        float(cube.shape[1])
    print(1 / high_cut)

    # Largest limit at 2 pix
    if high_cut > 0.25:
        high_cut = 0.25

    # Limit to ~1/5 of the box
    if ((1 / high_cut) / cube_size) > 0.1:
        high_cut = 1 / (0.1 * cube_size)

    # vca = VCA(cube).run(low_cut=1 / (32 * u.pix), high_cut=high_cut / u.pix,
    #                     verbose=True,
    #                     radial_pspec_kwargs={'binsize': 4.})

    # print(vca.slope, vca.slope2D, vca_exp)

    for sl_pix in pixs:
        if sl_pix == 1:
            cube_dsamp = cube
        else:
            cube_dsamp = cube.downsample_axis(sl_pix, axis=0)

        vca = VCA(cube_dsamp).run(low_cut=1 / (100 * u.pix),
                                  high_cut=(1 / 4.) / u.pix,
                                  fit_2D_kwargs={"fix_ellip_params": True},
                                  verbose=True,
                                  fit_2D=True)
        plt.draw()
        input((sl_pix, vca.slope, vca.slope2D, vca.ellip2D, vca.ellip2D_err))
        plt.clf()
        vca_slopes.append(vca.slope)
        vca_slopes_2D.append(vca.slope2D)
        vel_size.append(np.abs(np.diff(cube_dsamp.spectral_axis.value))[:1])

    # pspec = PowerSpectrum(mom0).run(low_cut=1 / (64 * u.pix),
    #                                 high_cut=0.5 / u.pix, verbose=False)
    pspec = PowerSpectrum(mom0).run(low_cut=1 / (100 * u.pix),
                                    high_cut=(1 / 4.) / u.pix,
                                    fit_2D_kwargs={"fix_ellip_params": True},
                                    verbose=False)

    outputs['VCA'].append(vca_slopes)
    outputs['VCA_2D'].append(vca_slopes_2D)

    outputs['pspec'].append(pspec.slope)
    outputs['pspec_2D'].append(pspec.slope2D)

    vel_size.append(np.ptp(cube.spectral_axis.value))

    plt.plot(vel_size,
             vca_slopes + [pspec.slope], 'o-', label='1D')
    plt.plot(vel_size,
             vca_slopes_2D + [pspec.slope2D], 'D--', label='2D')
    plt.axhline(- vca_exp, label='Thin Slice', alpha=0.5, linewidth=5)
    plt.axhline(- vca_thick_exp, label='Thick  Slice', alpha=0.5, linewidth=5)
    plt.axhline(- pspec_exp, label='Very Thick Slice', alpha=0.5, linewidth=5)
    plt.xlabel("Slice thickness (km/s)")
    plt.ylabel("Spatial Power Spectrum Slope")
    plt.legend(frameon=True)
    plt.draw()
    input(name)
    plt.clf()
