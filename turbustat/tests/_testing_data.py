
'''
Load in data sets for tests
'''

from ..io import fromfits
import numpy as np
keywords = {"centroid", "centroid_error", "integrated_intensity", "integrated_intensity_error", "linewidth",\
             "linewidth_error", "moment0", "moment0_error", "cube"}

folder1 = "/srv/astro/erickoch/enzo_sims/frac_factorial_set/Fiducial128_1.0.0/Fiducial128_1_21_0_0_flatrho_0021_13co/"

dataset1 = fromfits(folder1, keywords)

folder2 = "/srv/astro/erickoch/enzo_sims/frac_factorial_set/Design4.0.0/Design4_21_0_0_flatrho_0021_13co/"

dataset2 = fromfits(folder2, keywords)

computed_data = np.load("/srv/astro/caleb/checkVals.npz")

computed_distances = np.load("/srv/astro/caleb/computed_distances.npz")