# Licensed under an MIT open source license - see LICENSE


'''

Function for calculating statistics for sensitivity analysis

'''


import numpy as np
from turbustat.io import fromfits
from turbustat.statistics import stats_wrapper
import sys
import os
from datetime import datetime

keywords = {"centroid", "centroid_error", "integrated_intensity",
            "integrated_intensity_error", "linewidth",
            "linewidth_error", "moment0", "moment0_error", "cube"}
global keywords


def timestep_wrapper(fiducial_timestep, testing_timestep, statistics):

    fiducial_dataset = fromfits(fiducial_timestep, keywords)
    testing_dataset = fromfits(testing_timestep, keywords)

    distances = stats_wrapper(fiducial_dataset, testing_dataset,
                              statistics=statistics, multicore=True,
                              filenames=[fiducial_timestep, testing_timestep])
    return distances


def single_input(a):
    return timestep_wrapper(*a)


def run_all(fiducial, simulation_runs, face, statistics, savename,
            multicore=True, ncores=10, verbose=True):

    fiducial_timesteps = np.sort([os.path.join(fiducial, x)
                                  for x in os.listdir(fiducial)
                                  if os.path.isdir(os.path.join(fiducial, x))])

    timesteps_labels = [x[-8:] for x in fiducial_timesteps]

    if verbose:
        print "Simulation runs to be analyzed: %s" % (simulation_runs)
        print "Started at "+str(datetime.now())

    # Distances will be stored in an array of dimensions
    # # statistics x # sim runs x # timesteps
    # The +1 in the second dimensions is to include the
    # fiducial case against itself.
    distances_storage = np.zeros((len(statistics),
                                  len(simulation_runs),
                                  len(fiducial_timesteps)))

    for i, run in enumerate(simulation_runs):
        timesteps = np.sort([os.path.join(run, x) for x in os.listdir(run)
                             if os.path.isdir(os.path.join(run, x))])
        if verbose:
            print "On Simulation %s/%s" % (i+1, len(simulation_runs))
            print str(datetime.now())
        if multicore:
            pool = Pool(processes=ncores)
            distances = pool.map(single_input, izip(fiducial_timesteps,
                                                    timesteps,
                                                    repeat(statistics)))
            pool.close()
            pool.join()
            distances_storage[:, i, :] = \
                sort_distances(statistics, distances).T

        else:
            for ii, timestep in enumerate(timesteps):
                fiducial_dataset = fromfits(fiducial_timesteps[ii], keywords)
                testing_dataset = fromfits(timestep, keywords)
                if i == 0:
                    distances, fiducial_models = \
                        stats_wrapper(fiducial_dataset, testing_dataset,
                                      statistics=statistics)
                    all_fiducial_models = fiducial_models
                else:
                    distances = \
                        stats_wrapper(fiducial_dataset, testing_dataset,
                                      fiducial_models=all_fiducial_models,
                                      statistics=statistics)
                distances = [distances]
                distances_storage[:, i, ii:ii+1] = \
                    sort_distances(statistics, distances).T

    return distances_storage, timesteps_labels


def sort_distances(statistics, distances):
    if len(statistics) > 1:
        distance_array = np.empty((len(distances), len(statistics)))
    elif len(statistics) == 1:
        distance_array = np.empty((len(distances), 1))

    for j, dist in enumerate(distances):
        distance_array[j, :] = [distances[j][stat] for stat in statistics]

    return distance_array

if __name__ == "__main__":

    from multiprocessing import Pool
    from itertools import izip, repeat

    INTERACT = False  # Will prompt you for inputs if True
    PREFIX = "/srv/astro/erickoch/enzo_sims/frac_factorial_set/"

    os.chdir(PREFIX)

    statistics = ["DeltaVariance"]#"Wavelet", "MVC", "PSpec", "Bispectrum", "DeltaVariance",
                  # "Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF", "Cramer",
                  # "Skewness", "Kurtosis", "VCS_Density", "VCS_Velocity"]
                  #, "Dendrogram_Hist", "Dendrogram_Num", "PDF"]
    print "Statistics to run: %s" % (statistics)
    num_statistics = len(statistics)

    if INTERACT:
        fiducial = str(raw_input("Input folder of fiducial: "))
        face = str(raw_input("Which face? (0 or 2): "))
        save_name = str(raw_input("Save Name: "))
        MULTICORE = bool(raw_input("Run on multiple cores? (T or blank): "))

        if MULTICORE:
            NCORES = int(raw_input("How many cores to use? "))
    else:
        fiducial = str(sys.argv[1])
        face = str(sys.argv[2])
        save_name = str(sys.argv[3])
        MULTICORE = str(sys.argv[4])
        if MULTICORE == "T":
            MULTICORE = True
        else:
            MULTICORE = False
            NCORES = 1  # Placeholder to pass into run_all
        if MULTICORE:
            NCORES = int(sys.argv[5])

    if fiducial == "fid_comp":  # Run all the comparisons of fiducials
        if INTERACT:
            cross_comp = str(raw_input("Cross comparison? "))
        else:
            cross_comp = str(sys.argv[6])

        if cross_comp == "F":
            cross_comp = False
        else:
            cross_comp = True

        if cross_comp:
            if face == "0":
                comp_face = "2"
            elif face == "2":
                comp_face = "0"
        else:
            comp_face = face

        fiducials = [x for x in os.listdir(".") if os.path.isdir(x)
                     and x[:8] == "Fiducial" and x[-3] == comp_face]
        fiducials = np.sort(fiducials)

        fiducials_comp = [x for x in os.listdir(".") if os.path.isdir(x)
                          and x[:8] == "Fiducial" and x[-3] == face]
        fiducials_comp = np.sort(fiducials_comp)

        print "Fiducials to compare %s" % (fiducials)
        fiducial_labels = []
        # number of comparisons b/w all fiducials
        num_comp = (len(fiducials)**2. - len(fiducials))/2
        # Change dim 2 to match number of time steps
        distances_storage = np.zeros((num_statistics, num_comp, 10))
        posn = 0
        prev = 0
        # no need to loop over the last one
        for fid, i in zip(fiducials[:-1], np.arange(len(fiducials)-1, 0, -1)):
            fid_num = int(fid[-5])#+1 #### THIS NEED TO BE CHANGED BASED ON THE FIDUCIAL NUMBERING!!!!!!!
            posn += i
            partial_distances, timesteps_labels = \
                run_all(fiducials[fid_num-1], fiducials_comp[fid_num:],
                        face, statistics, save_name, multicore=MULTICORE,
                        ncores=NCORES)
            distances_storage[:, prev:posn, :] = partial_distances
            prev += i
            fiducial_labels.extend([f + "to" + fid for f in
                                    fiducials_comp[fid_num:]])

        # consistent naming with non-fiducial case
        simulation_runs = fiducial_labels
        face = comp_face
    else:  # Normal case of comparing to single fiducial

        simulation_runs = [x for x in os.listdir(".") if os.path.isdir(x)
                           and x[:6] == "Design" and x[-3] == face]
        simulation_runs = np.sort(simulation_runs)

        distances_storage, timesteps_labels = \
            run_all(fiducial, simulation_runs,
                    face, statistics, save_name,
                    multicore=MULTICORE, ncores=NCORES)

        simulation_runs = [sim+"to"+fiducial for sim in simulation_runs]

    filename = save_name + "_" + face + "_distance_results.h5"
    print filename
    from pandas import DataFrame, HDFStore, concat

    ## Save data for each statistic in a dataframe.
    ## Each dataframe is saved in a single hdf5 file

    store = HDFStore("results/"+filename)

    for i in range(num_statistics):
        df = DataFrame(distances_storage[i, :, :], index=simulation_runs,
                       columns=timesteps_labels)
        if statistics[i] in store:
            existing_df = store[statistics[i]]
            if len(existing_df.index) == len(df.index):
                store[statistics[i]] = df
            else:  # Append on
                for ind in df.index:
                    if ind in list(existing_df.index):
                        existing_df.ix[ind] = df.ix[ind]
                    else:
                        existing_df = concat([existing_df, df])
                    store[statistics[i]] = existing_df
        else:
            store[statistics[i]] = df

    store.close()

    print "Done at " + str(datetime.now())
