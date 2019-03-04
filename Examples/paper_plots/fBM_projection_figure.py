
'''
Show zeroth moments of the 4 example cubes.
'''


from spectral_cube import SpectralCube
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb

col_pal = sb.color_palette()

# data_path = "/Volumes/Travel_Data/Turbulence/fBM_cubes/"
data_path = os.path.expanduser("~/MyRAID/Astrostat/TurbuStat_Paper/fBM_cubes/")

reps = range(4)
cube_size = 256

dens = 4
vel = 4

markers = ['D', 'o', 's', 'p', '*']

width = 8.4
# Keep the default ratio used in seaborn. This can get overwritten.
height = (4.4 / 6.4) * width

figsize = (width, height)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                         figsize=figsize)

mom0s = []

for rep in reps:

    name = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}"\
        .format(np.abs(dens), np.abs(vel), rep)

    filename = "fBM_density_{0:.2f}_velocity_{1:.2f}_rep_{2}_size_{3}.fits"\
        .format(np.abs(dens), np.abs(vel), rep, cube_size)

    cube = SpectralCube.read(os.path.join(data_path, filename))

    mom0s.append(cube.moment0())

max_val = max([mom0.max().value for mom0 in mom0s])
min_val = min([mom0.min().value for mom0 in mom0s])

for i, ax in enumerate(axes.ravel()):
    im = ax.imshow(mom0s[i].value, origin='lower', vmax=max_val,
                   vmin=min_val, cmap='viridis')

cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.set_label("Integrated Intensity (K km/s)")

plt.subplots_adjust(wspace=0.00, hspace=0.06, right=0.75,
                    left=0.21)

plt.savefig("../figures/cube_moment0s.png")
plt.savefig("../figures/cube_moment0s.pdf")
plt.close()
