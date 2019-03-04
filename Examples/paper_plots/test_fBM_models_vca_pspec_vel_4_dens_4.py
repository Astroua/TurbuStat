
'''
Example VCA and Pspec for the fBM cubes that suffer the least from shot noise.
These are the cubes with velocity and density indices of -4.
'''

from spectral_cube import SpectralCube
from astropy.table import Table, Column
import numpy as np
import astropy.units as u
from astropy import constants as cc
import os
import matplotlib.pyplot as plt
import seaborn as sb

from turbustat.statistics import PowerSpectrum, VCA, SCF, PCA

col_pal = sb.color_palette()

plt.rcParams['axes.unicode_minus'] = False

# data_path = "/Volumes/Travel_Data/Turbulence/fBM_cubes/"
data_path = os.path.expanduser("~/MyRAID/Astrostat/TurbuStat_Paper/fBM_cubes/")

reps = range(4)
cube_size = 256

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


dens = 4
vel = 4

markers = ['D', 'o', 's', 'p', '*']

width = 8.75
# fig_ratio = (4.4 / 6.4) / 2
height = 5.07
figsize = (width, height)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True,
                         figsize=figsize)

for rep in reps:

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
                                  verbose=False,
                                  fit_2D=True)
        # plt.draw()
        # input((sl_pix, vca.slope, vca.slope2D, vca.ellip2D, vca.ellip2D_err))
        # plt.clf()
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

    axes[0].semilogx(vel_size,
                     vca_slopes + [pspec.slope], '-',
                     label='{}'.format(rep + 1),
                     marker=markers[rep])

    axes[1].semilogx(vel_size,
                     vca_slopes_2D + [pspec.slope2D], '-',
                     marker=markers[rep])

axes[0].axhline(-vca_exp, alpha=0.5, linewidth=5,
                color='k', zorder=-10)
axes[0].axhline(-vca_thick_exp, alpha=0.5, linewidth=5,
                color='k', zorder=-10, linestyle='-')
axes[0].axhline(-pspec_exp, alpha=0.5, linewidth=5,
                color='k', zorder=-10, linestyle='-')

axes[0].annotate("Thin Slice", xy=(10., -2.5),
                 bbox={"boxstyle": "round", "facecolor": "w",
                       "edgecolor": 'k'},
                 horizontalalignment='left',
                 verticalalignment='center')
axes[0].annotate("Thick Slice", xy=(0.2, -3.5),
                 bbox={"boxstyle": "round", "facecolor": "w"},
                 horizontalalignment='left',
                 verticalalignment='center')
axes[0].annotate("Very Thick Slice", xy=(0.2, -4),
                 bbox={"boxstyle": "round", "facecolor": "w"},
                 horizontalalignment='left',
                 verticalalignment='center')

axes[1].axhline(-vca_exp, alpha=0.5, linewidth=5,
                color='k', zorder=-10)
axes[1].axhline(-vca_thick_exp, alpha=0.5, linewidth=5,
                color='k', zorder=-10, linestyle='-')
axes[1].axhline(-pspec_exp, alpha=0.5, linewidth=5,
                color='k', zorder=-10, linestyle='-')

axes[0].axvline(v_th.value, alpha=0.9, color=col_pal[5], zorder=-1,
                linestyle='--', linewidth=5)
axes[1].axvline(v_th.value, alpha=0.9, color=col_pal[5], zorder=-1,
                linestyle='--', linewidth=5)

axes[0].set_ylabel("Power Spectrum Index")

# Change to figure text in centre
fig.text(0.5, 0.04, "Slice thickness (km/s)", ha='center')

axes[0].set_title("1D Fit")
axes[1].set_title("2D Fit")

fig.legend(frameon=True, loc=(0.38, 0.5), framealpha=0.9)

axes[0].grid()
axes[1].grid()

axes[0].set_ylim([-4.1, -2.4])

# plt.tight_layout()
plt.subplots_adjust(wspace=0.02, bottom=0.18)

plt.savefig("../figures/vca_slice_thickness_recovery.png")
plt.savefig("../figures/vca_slice_thickness_recovery.pdf")
plt.close()
