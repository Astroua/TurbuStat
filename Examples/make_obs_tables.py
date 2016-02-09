
# Create data tables of the observational results
# Run from Dropbox/AstroStatistics/Full Factorial/Observational Results/

import os
import shutil

from turbustat.analysis.convert_results import concat_convert_HDF5
from turbustat.analysis import convert_format


# Obs to Fids

path = "Obs_to_Fid/"
hdf5_path = "Obs_to_Fid/HDF5/"

convert_format(hdf5_path, 0)
shutil.move(hdf5_path+"complete_distances_face_0.csv", path)

convert_format(hdf5_path, 1)
shutil.move(hdf5_path+"complete_distances_face_1.csv", path)

convert_format(hdf5_path, 2)
shutil.move(hdf5_path+"complete_distances_face_2.csv", path)

# Des to Obs

hdf5_path = "Des_to_Obs/HDF5/"

concat_convert_HDF5(hdf5_path, face=0, interweave=True, average_axis=0)
shutil.move(os.path.join(hdf5_path, "distances_0.csv"), "Des_to_Obs")
shutil.move("Des_to_Obs/distances_0.csv", "Des_to_Obs/distances_0_obs.csv")

concat_convert_HDF5(hdf5_path, face=2, interweave=True, average_axis=0)
shutil.move(os.path.join(hdf5_path, "distances_2.csv"), "Des_to_Obs")
shutil.move("Des_to_Obs/distances_2.csv", "Des_to_Obs/distances_2_obs.csv")

# Make plots using make_distance_plots.py
