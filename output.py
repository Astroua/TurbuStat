
'''

Function for calculating statistics for sensitivity analysis

'''


import numpy as np
from utilities import fromfits
import sys
import os
from datetime import datetime

keywords = {"centroid", "centroid_error", "integrated_intensity", "integrated_intensity_error", "linewidth",\
             "linewidth_error", "moment0", "moment0_error", "cube"}

## Load each statistic in

from wavelets import Wavelet_Distance

from mvc import MVC_distance

from pspec_bispec import PSpec_Distance, BiSpectrum_Distance

from genus import GenusDistance

from delta_variance import DeltaVariance_Distance

from vca_vcs import VCA_Distance, VCS_Distance

from tsallis import Tsallis_Distance

from stat_moments import StatMomentsDistance

from pca import PCA_Distance

from scf import SCF_Distance

from cramer import Cramer_Distance

## Wrapper function
def wrapper(dataset1, dataset2, fiducial_models=None, statistics=None, multicore=False):

    if statistics is None: #Run them all
        statistics = ["Wavelet", "MVC", "PSpec", "Bispectrum","DeltaVariance","Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF",
                   "Cramer","Skewness", "Kurtosis"]
    if any("Skewness" in s for s in statistics):
        # There will be an indexing
        # issue without this. This the lazy fix.
        index = statistics.index("Skewness")
        statistics.insert(len(statistics), statistics.pop(index))

    distances = []

    if fiducial_models is None: # Calculate the fiducial case and return it for later use

        fiducial_models = {}

        if any("Wavelet" in s for s in statistics):
            wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric()
            distances.append(wavelet_distance.distance)
            fiducial_models["Wavelet"] = wavelet_distance.wt1

        if any("MVC" in s for s in statistics):
            mvc_distance = MVC_distance(dataset1, dataset2).distance_metric()
            distances.append(mvc_distance.distance)
            fiducial_models["MVC"] = mvc_distance.mvc1

        if any("PSpec" in s for s in statistics):
            pspec_distance = PSpec_Distance(dataset1, dataset2).distance_metric()
            distances.append(pspec_distance.distance)
            fiducial_models["PSpec"] = pspec_distance.pspec1

        if any("Bispectrum" in s for s in statistics):
            bispec_distance = BiSpectrum_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric()
            distances.append(bispec_distance.distance)
            fiducial_models["Bispectrum"] = bispec_distance.bispec1

        if any("DeltaVariance" in s for s in statistics):
            delvar_distance = DeltaVariance_Distance(dataset1["integrated_intensity"][0], dataset1["integrated_intensity_error"][0], \
                                            dataset2["integrated_intensity"][0], dataset2["integrated_intensity_error"][0]).distance_metric()
            distances.append(delvar_distance.distance)
            fiducial_models["DeltaVariance"] = delvar_distance.delvar1

        if any("Genus" in s for s in statistics):
            genus_distance = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric()
            distances.append(genus_distance.distance)
            fiducial_models["Genus"] = genus_distance.genus1

        if any("VCS" in s for s in statistics):
            vcs_distance = VCS_Distance(dataset1["cube"],dataset2["cube"]).distance_metric()
            distances.append(vcs_distance.distance)
            fiducial_models["VCS"] = vcs_distance.vcs1

        if any("VCA" in s for s in statistics):
            vca_distance = VCA_Distance(dataset1["cube"],dataset2["cube"]).distance_metric()
            distances.append(vca_distance.distance)
            fiducial_models["VCA"] = vca_distance.vca1

        if any("Tsallis" in s for s in statistics):
            tsallis_distance= Tsallis_Distance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric()
            distances.append(tsallis_distance.distance)
            fiducial_models["Tsallis"] = tsallis_distance.tsallis1

        if any("Skewness" in s for s in statistics) or any("Kurtosis" in s for s in statistics):
            moment_distance = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5).distance_metric()
            distances.append(moment_distance.skewness_distance)
            distances.append(moment_distance.kurtosis_distance)
            fiducial_models["stat_moments"] = moment_distance.moments1

        if any("PCA" in s for s in statistics):
            pca_distance = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric()
            distances.append(pca_distance.distance)
            fiducial_models["PCA"] = pca_distance.pca1

        if any("SCF" in s for s in statistics):
            scf_distance = SCF_Distance(dataset1["cube"][0],dataset2["cube"][0]).distance_metric()
            distances.append(scf_distance.distance)
            fiducial_models["SCF"] = scf_distance.scf1

        if any("Cramer" in s for s in statistics):
            cramer_distance = Cramer_Distance(dataset1["cube"][0], dataset2["cube"][0]).distance_metric()
            distances.append(cramer_distance.distance)

        distances = np.asarray(distances)

        if multicore:
            return distances
        else:
            return distances, fiducial_models

    else:

        if any("Wavelet" in s for s in statistics):
            wavelet_distance = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"],
                fiducial_model=fiducial_models["Wavelet"]).distance_metric()
            distances.append(wavelet_distance.distance)

        if any("MVC" in s for s in statistics):
            mvc_distance = MVC_distance(dataset1, dataset2, fiducial_model=fiducial_models["MVC"]).distance_metric()
            distances.append(mvc_distance.distance)

        if any("PSpec" in s for s in statistics):
            pspec_distance = PSpec_Distance(dataset1, dataset2, fiducial_model=fiducial_models["PSpec"]).distance_metric()
            distances.append(pspec_distance.distance)

        if any("Bispectrum" in s for s in statistics):
            bispec_distance = BiSpectrum_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"],
                fiducial_model=fiducial_models["Bispectrum"]).distance_metric()
            distances.append(bispec_distance.distance)

        if any("DeltaVariance" in s for s in statistics):
            delvar_distance = DeltaVariance_Distance(dataset1["integrated_intensity"][0], dataset1["integrated_intensity_error"][0], \
                                            dataset2["integrated_intensity"][0], dataset2["integrated_intensity_error"][0],
                                            fiducial_model=fiducial_models["DeltaVariance"]).distance_metric()
            distances.append(delvar_distance.distance)

        if any("Genus" in s for s in statistics):
            genus_distance = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0],
                fiducial_model=fiducial_models["Genus"]).distance_metric()
            distances.append(genus_distance.distance)

        if any("VCS" in s for s in statistics):
            vcs_distance = VCS_Distance(dataset1["cube"],dataset2["cube"],
                fiducial_model=fiducial_models["VCS"]).distance_metric()
            distances.append(vcs_distance.distance)

        if any("VCA" in s for s in statistics):
            vca_distance = VCA_Distance(dataset1["cube"],dataset2["cube"],
                fiducial_model=fiducial_models["VCA"]).distance_metric()
            distances.append(vca_distance.distance)

        if any("Tsallis" in s for s in statistics):
            tsallis_distance= Tsallis_Distance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0],
                fiducial_model=fiducial_models["Tsallis"]).distance_metric()
            distances.append(tsallis_distance.distance)

        if any("Skewness" in s for s in statistics) or any("Kurtosis" in s for s in statistics):
            moment_distance = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5,
                fiducial_model=fiducial_models["stat_moments"]).distance_metric()
            distances.append(moment_distance.skewness_distance)
            distances.append(moment_distance.kurtosis_distance)

        if any("PCA" in s for s in statistics):
            pca_distance = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0],
                fiducial_model=fiducial_models["PCA"]).distance_metric()
            distances.append(pca_distance.distance)

        if any("SCF" in s for s in statistics):
            scf_distance = SCF_Distance(dataset1["cube"][0],dataset2["cube"][0],
                fiducial_model=fiducial_models["SCF"]).distance_metric()
            distances.append(scf_distance.distance)

        if any("Cramer" in s for s in statistics):
            cramer_distance = Cramer_Distance(dataset1["cube"][0], dataset2["cube"][0]).distance_metric()
            distances.append(cramer_distance.distance)

        return np.asarray(distances)

def timestep_wrapper(fiducial_timestep, testing_timestep):
    keywords = {"centroid", "centroid_error", "integrated_intensity", "integrated_intensity_error", "linewidth",\
             "linewidth_error", "moment0", "moment0_error", "cube"}
    statistics = ["Wavelet", "MVC", "PSpec", "Bispectrum","DeltaVariance","Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF",
                   "Cramer","Skewness", "Kurtosis"]
    fiducial_dataset = fromfits(fiducial_timestep, keywords)
    testing_dataset = fromfits(testing_timestep, keywords)

    distances = wrapper(fiducial_dataset, testing_dataset, statistics=statistics,
        multicore=True)
    return distances

def single_input(a):
    return timestep_wrapper(*a)

if __name__ == "__main__":

    from multiprocessing import Pool
    from itertools import izip, repeat

    INTERACT = True ## Will prompt you for inputs if True
    PREFIX = "/srv/astro/erickoch/enzo_sims/"

    os.chdir(PREFIX)

    statistics = ["Wavelet", "MVC", "PSpec", "Bispectrum","DeltaVariance","Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF",
                  "Cramer", "Skewness", "Kurtosis"]
    print statistics
    num_statistics = len(statistics)

    if INTERACT:
        fiducial = str(raw_input("Input folder of fiducial: "))
        face = str(raw_input("Which face? (0 or 2): "))
        save_name = str(raw_input("Save Name: "))
        MULTICORE = bool(raw_input("Run on multiplecores? (T or blank): "))

        if MULTICORE :
            NCORES = int(raw_input("How many cores to use? "))
    else:
        fiducial = str(sys.argv[1])
        face = str(sys.argv[2])
        save_name = str(sys.argv[3])
        MULTICORE = bool(sys.argv[4])
        if MULTICORE:
            NCORES = int(sys.argv[5])

    fiducial_timesteps = [os.path.join(fiducial,x) for x in os.listdir(fiducial) if os.path.isdir(os.path.join(fiducial,x))]
    timesteps_labels = [x[-8:] for x in fiducial_timesteps]

    simulation_runs = ["Fiducial128_4.2.0"]#[x for x in os.listdir(".") if os.path.isdir(x) and x[:6]=="Design" and x[-3]==face]
    # simulation_runs.remove("hd22_arrays")

    print "Simulation runs to be analyzed: %s" % (simulation_runs)
    print "Started at "+str(datetime.now())

    ## Distances will be stored in an array of dimensions # statistics x # sim runs x # timesteps
    ## The +1 in the second dimensions is to include the fiducial case against itself
    distances_storage = np.zeros((num_statistics, len(simulation_runs), len(fiducial_timesteps)))

    for i, run in enumerate(simulation_runs):
        timesteps = [os.path.join(run,x) for x in os.listdir(run) if os.path.isdir(os.path.join(run,x))]

        print "On Simulation %s/%s" % (i+1,len(simulation_runs))
        print str(datetime.now())
        if MULTICORE:
            pool = Pool(processes=NCORES)
            distances = pool.map(single_input, izip(fiducial_timesteps, timesteps))
            distances_storage[:,i,:] = np.asarray(distances).T
            pool.close()
            pool.join()
        else:
            for ii, timestep in enumerate(timesteps):
                fiducial_dataset = fromfits(fiducial_timesteps[ii], keywords)
                testing_dataset = fromfits(timestep, keywords)
                if i==0:
                    distances, fiducial_models = wrapper(fiducial_dataset, testing_dataset,
                        statistics=statistics)
                    all_fiducial_models = fiducial_models
                else:
                    distances = wrapper(fiducial_dataset, testing_dataset, fiducial_models=all_fiducial_models,
                        statistics=statistics)
                print distances
                distances_storage[:,i,ii] = distances

    # simulation_runs.insert(0, fiducial)
    ## Save data for each statistic in a dataframe. Each dataframe is saved in a single hdf5 file
    from pandas import DataFrame, HDFStore

    store = HDFStore(save_name+"_"+face+"_distance_results.h5")
    simulation_runs = [sim+"to"+fiducial for sim in simulation_runs]
    for i in range(num_statistics):
        df = DataFrame(distances_storage[i,:,:], index=simulation_runs, columns=timesteps_labels)
        store[statistics[i]] = df

    store.close()

    print "Done at "+str(datetime.now())
