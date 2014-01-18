
'''

Function for calculating statistics for sensitivity analysis

'''

INTERACT = True ## Will prompt you for inputs if True
PREFIX = "/srv/astro/erickoch/simcloud_stats/testing_folder"

import numpy as np
from utilities import fromfits
import sys
import os

keywords = {"centroid", "centroid_error", "integrated_intensity", "integrated_intensity_error", "linewidth",\
             "linewidth_error", "moment0", "moment0_error", "cube"}
statistics = ["Wavelet", "MVC", "PSpec", "Genus", "VCS", "VCA", "Tsallis", "Skewness", "Kurtosis",
              "PCA", "SCF"]

## Load each statistic in

from wavelets import Wavelet_Distance

from mvc import MVC_distance

from pspec_bispec import PSpec_Distance, BiSpec_Distance

from genus import GenusDistance

from delta_variance import DeltaVariance_Distance

from vca_vcs import VCA_Distance, VCS_Distance

from tsallis import Tsallis_Distance

from stat_moments import StatMomentsDistance

from pca import PCA_Distance

from scf import SCF_Distance

os.chdir(PREFIX)

## Wrapper function
def wrapper(dataset1, dataset2, fiducial=False, fiducial_models=False):

    if fiducial: # Calculate the fiducial case and return it for later use
        wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric()
        mvc_distance = MVC_distance(dataset1, dataset2).distance_metric()
        pspec_distance = PSpec_Distance(dataset1, dataset2).distance_metric()
        # bispec_distance = BiSpec_Distance()
        delvar_distance = DeltaVariance_Distance(dataset1["integrated_intensity"][0], dataset1["integrated_intensity_error"][0], \
                                            dataset2["integrated_intensity"][0], dataset2["integrated_intensity_error"][0]).distance_metric()
        genus_distance = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric()
        vcs_distance = VCS_Distance(dataset1["cube"],dataset2["cube"]).distance_metric()
        vca_distance = VCA_Distance(dataset1["cube"],dataset2["cube"]).distance_metric()
        tsallis_distance= Tsallis_Distance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric()
        moment_distance = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5).distance_metric()
        pca_distance = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric()
        scf_distance = SCF_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric()

        distances = np.asarray([ wavelet_distance.distance, mvc_distance.distance, pspec_distance.distance, # bispec_distance.distance, delvar_distance.distance, \
                                genus_distance.distance, vcs_distance.distance, vca_distance.distance, tsallis_distance.distance, \
                                moment_distance.kurtosis_distance, moment_distance.skewness_distance])

        ## NEED TO ADD BISPEC AND DELVAR STILL
        fiducial_models = [ wavelet_distance.wt1, mvc_distance.mvc1, pspec_distance.pspec1, genus_distance.genus1,\
                           vcs_distance.vcs1, vca_distance.vca1, tsallis_distance.tsallis1, moment_distance.moments1]

        return distances, fiducial_models

    else:
        if not fiducial_models:
            raise ValueError("Must provide fiducial models to run non-fiducial case.")

        wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"], fiducial_model=fiducial_models[0]).distance_metric()

        mvc_distance = MVC_distance(dataset1, dataset2, fiducial_model=fiducial_models[1]).distance_metric()

        pspec_distance = PSpec_Distance(dataset1, dataset2, fiducial_model=fiducial_models[2]).distance_metric()

        # bispec_distance = BiSpec_Distance(, fiducial_model=fiducial_models[3])

        delvar_distance = DeltaVariance_Distance(dataset1["integrated_intensity"][0], dataset1["integrated_intensity_error"][0], \
                                            dataset2["integrated_intensity"][0], dataset2["integrated_intensity_error"][0],
                                            fiducial=fiducial_models[3]).distance_metric()

        genus_distance = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0],
                                            fiducial_model=fiducial_models[4]).distance_metric()

        vcs_distance = VCS_Distance(dataset1["cube"],dataset2["cube"], fiducial_model=fiducial_models[5]).distance_metric()

        vca_distance = VCA_Distance(dataset1["cube"],dataset2["cube"], fiducial_model=fiducial_models[6]).distance_metric()

        tsallis_distance= Tsallis_Distance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0],
                                            fiducial_model=fiducial_models[7]).distance_metric()

        moment_distance = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5,
                                            fiducial_model=fiducial_models[8]).distance_metric()

        pca_distance = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0], fiducial=fiducial_models[9]).distance_metric()

        scf_distance = SCF_Distance(dataset1["cube"][0],dataset2["cube"][0], fiducial=fiducial_models[10]).distance_metric()

        return np.asarray([wavelet_distance.distance, mvc_distance.distance, pspec_distance.distance,  delvar_distance.distance,# bispec_distance.distance, \
                                genus_distance.distance, vcs_distance.distance, vca_distance.distance, tsallis_distance.distance, \
                                moment_distance.kurtosis_distance, moment_distance.skewness_distance,
                                pca_distance.distance, scf_distance.distance])


if INTERACT:
    fiducial = str(raw_input("Input folder of fiducial: "))
    num_statistics = int(raw_input("Number of Statistics? "))
else:
    fiducial = str(sys.argv[1])
    num_statistics = int(sys.argv[2])

fiducial_timesteps = [os.path.join(fiducial,x) for x in os.listdir(fiducial) if os.path.isdir(os.path.join(fiducial,x))]
timesteps_labels = [x[-8:] for x in fiducial_timesteps]

simulation_runs = [x for x in os.listdir(".") if os.path.isdir(x) and x!=fiducial]
# simulation_runs.remove("hd22_arrays")
print "Simulation runs to be analyzed: %s" % (simulation_runs)
## Distances will be stored in an array of dimensions # statistics x # sim runs x # timesteps
# The +1 in the second dimensions is to include the fiducial case against itself
distances_storage = np.zeros((num_statistics, len(simulation_runs)+1, len(fiducial_timesteps)))

for i, run in enumerate(simulation_runs):
    timesteps = [os.path.join(run,x) for x in os.listdir(run) if os.path.isdir(os.path.join(run,x))]
    print i
    for ii, timestep in enumerate(timesteps):
        fiducial_dataset = fromfits(fiducial_timesteps[ii], keywords)
        testing_dataset = fromfits(timestep, keywords)
        if i==0:
            distances, fiducial_models = wrapper(fiducial_dataset, testing_dataset, fiducial=True)
            all_fiducial_models = fiducial_models
        else:
            distances = wrapper(fiducial_dataset, testing_dataset, fiducial_models=all_fiducial_models)
        distances_storage[:,i+1,ii] = distances

simulation_runs.insert(0, fiducial)
## Save data for each statistic in a dataframe. Each dataframe is saved in a single hdf5 file
from pandas import DataFrame, HDFStore

store = HDFStore("distance_results.h5")

for i in range(num_statistics):
    df = DataFrame(distances_storage[i,:,:], index=simulation_runs, columns=timesteps_labels)
    store[statistics[i]] = df

store.close()

