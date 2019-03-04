
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

from turbustat.statistics import PCA

font_scale = 1.25

width = 4.2
# Keep the default ratio used in seaborn. This can get overwritten.
height = (4.4 / 6.4) * width
figsize = (width, height)

sb.set_context("paper", font_scale=font_scale,
               rc={"figure.figsize": figsize})
sb.set_palette("colorblind")

col_pal = sb.color_palette()

# data_path = "/Volumes/Travel_Data/Turbulence/fBM_cubes/"
data_path = os.path.expanduser("~/MyRAID/Astrostat/TurbuStat_Paper/fBM_cubes/")

reps = range(4)
cube_size = 256

dens = 4
vel = 4

markers = ['D', 'o', 's', 'p', '*']

fig = plt.figure(figsize=figsize)

# Now run PCA

for rep in reps:

    name = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}"\
        .format(np.abs(dens), np.abs(vel), rep)

    filename = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}_size_{3}.fits"\
        .format(np.abs(dens), np.abs(vel), rep, cube_size)

    cube = SpectralCube.read(os.path.join(data_path, filename))
    cube.allow_huge_operations = True

    pca = PCA(cube).run(min_eigval=5e-3)

    # Grab spatial and spectral scales
    spat_scale = pca.spatial_width(u.pix)
    spat_err_scale = pca.spatial_width_error(u.pix)
    spec_scale = pca.spectral_width(u.km / u.s)
    spec_err_scale = pca.spectral_width_error(u.km / u.s)

    plt.errorbar(np.log10(spat_scale.value),
                 np.log10(spec_scale.value),
                 xerr=0.434 * spat_err_scale.value / spat_scale.value,
                 yerr=0.434 * spec_err_scale.value / spec_scale.value,
                 marker=markers[rep],
                 linestyle='none', color=col_pal[rep],
                 label=r"{0}: $\alpha={1:.2f}^{{+{2:.2f}}}_{{-{3:.2f}}}$"
                        .format(rep + 1, pca.index,
                                pca.index_error_range[1] - pca.index,
                                pca.index - pca.index_error_range[0]))
    xvals_pix = \
        np.linspace(np.log10(np.nanmin(pca.spatial_width(u.pix).value)),
                    np.log10(np.nanmax(pca.spatial_width(u.pix).value)),
                    spat_scale.size * 10)
    xvals = \
        np.linspace(np.log10(np.nanmin(pca.spatial_width(u.pix).value)),
                    np.log10(np.nanmax(pca.spatial_width(u.pix).value)),
                    spat_scale.size * 10)

    intercept = pca.intercept(unit=u.pix)

    spec_conv = pca._to_spectral(1 * u.pix, u.km / u.s).value

    plt.plot(xvals,
             np.log10(10**(pca.index * xvals_pix +
                           np.log10(intercept.value)) * spec_conv),
             '-', color=col_pal[rep], alpha=0.7)


plt.grid()
plt.xlabel("log Spatial Size / pix")
plt.ylabel("log Spectral Size / (km / s)")
plt.legend(frameon=True, ncol=2)
plt.ylim([-0.1, 1.17])
plt.tight_layout()

plt.savefig("../figures/pca_size_lwidth_recovery.png")
plt.savefig("../figures/pca_size_lwidth_recovery.pdf")
plt.close()
