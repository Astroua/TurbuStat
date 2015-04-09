# Licensed under an MIT open source license - see LICENSE


'''

Function for calculating statistics for sensitivity analysis

'''


import numpy as np
import sys
import os
import copy
from datetime import datetime
from astropy.io.fits import getdata

from turbustat.statistics import stats_wrapper
from turbustat.data_reduction import property_arrays


keywords = {"centroid", "centroid_error", "integrated_intensity",
            "integrated_intensity_error", "linewidth",
            "linewidth_error", "moment0", "moment0_error", "cube"}


def timestep_wrapper(fiducial_timestep, testing_timestep, statistics,
                     add_noise, rms_noise):

    # Derive the property arrays assuming uniform noise (for sims)
    fiducial_dataset = load_and_reduce(fiducial_timestep, add_noise=add_noise,
                                       rms_noise=rms_noise)
    testing_dataset = load_and_reduce(testing_timestep, add_noise=add_noise,
                                      rms_noise=rms_noise)

    if add_noise:
        vca_break = 1.5
        vcs_break = -0.5
    else:
        vca_break = None
        vcs_break = -0.8

    distances = stats_wrapper(fiducial_dataset, testing_dataset,
                              statistics=statistics, multicore=True,
                              vca_break=vca_break, vcs_break=vcs_break)
    return distances


def single_input(a):
    return timestep_wrapper(*a)


def run_all(fiducial, simulation_runs, face, statistics, savename,
            multicore=True, ncores=10, verbose=True, comp_face=None,
            multi_timesteps=False):
    '''
    Given a fiducial set and a series of sets to compare to, loop
    through and compare all sets and their timesteps. Return an array of
    the distances.

    Parameters
    ----------
    verbose : bool, optional
        Prints out the time when completing a set.
    comp_face : int or None, optional
        Face to compare to. If None, this is set to the fiducial face.
    multi_timesteps : bool, optional
        If multiple timesteps are given for each simulation run, parallelize
        over the timesteps. If only one is given, parallelize over the
        simulation runs.
    '''
    if comp_face is None:
        comp_face = face

    fiducial_timesteps = fiducial[face]

    if verbose:
        # print "Simulation runs to be analyzed: %s" % (simulation_runs)
        print "Started at "+str(datetime.now())

    if multi_timesteps:
        # Distances will be stored in an array of dimensions
        # # statistics x # sim runs x # timesteps
        distances_storage = np.zeros((len(statistics),
                                      len(simulation_runs),
                                      len(fiducial_timesteps)))

        print distances_storage.shape

        for i, key in enumerate(simulation_runs.keys()):
            timesteps = simulation_runs[key][comp_face]

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
                    fiducial_dataset = load_and_reduce(fiducial_timesteps[ii])
                    testing_dataset = load_and_reduce(timestep)
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

    else:
        distances_storage = np.zeros((len(statistics),
                                      len(simulation_runs)))

        if multicore:
            pool = Pool(processes=ncores)
            distances = pool.map(single_input, izip(repeat(fiducial),
                                                    simulation_runs,
                                                    repeat(statistics)))
            pool.close()
            pool.join()
            distances_storage = sort_distances(statistics, distances).T

        else:
            for i, key in enumerate(simulation_runs.keys()):
                fiducial_dataset = load_and_reduce(fiducial[face])
                testing_dataset = \
                    load_and_reduce(simulation_runs[key][comp_face])
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
                distances_storage[:, i:i+1] = \
                    sort_distances(statistics, distances).T

    return distances_storage


def sort_distances(statistics, distances):
    if len(statistics) > 1:
        distance_array = np.empty((len(distances), len(statistics)))
    elif len(statistics) == 1:
        distance_array = np.empty((len(distances), 1))

    for j, dist in enumerate(distances):
        distance_array[j, :] = [distances[j][stat] for stat in statistics]

    return distance_array


def files_sorter(folder, fiducial_labels=np.arange(0, 5, 1),
                 design_labels=np.arange(0, 32, 1), timesteps='last',
                 faces=[0, 1, 2], suffix="fits", append_prefix=False):
    '''
    If the entire simulation suite is in one directory, this function
    will spit out appropriate groupings.

    Parameters
    ----------
    folder : str
        Folder where data is.
    fiducial_labels : list or numpy.ndarray, optional
        List of the fiducial numbers.
    design_labels : list or numpy.ndarray, optional
        List of the design numbers.
    timesteps : 'last' or list or numpy.ndarray, optional
        List of timesteps to analyze. If 'last', the last timestep
        found for each simulation is used.
    faces : list
        Faces of the simulations to use.
    suffix : str, optional
        File suffix.
    '''

    # Get the files and remove any sub-directories.
    files = [f for f in os.listdir(folder) if not os.path.isdir(f) and
             f[-len(suffix):] == suffix]

    # Set up the dictionaries.
    fiducials = dict.fromkeys(fiducial_labels)
    for lab in fiducial_labels:
        fiducials[lab] = dict((face, []) for face in faces)
    designs = dict.fromkeys(design_labels)
    for lab in design_labels:
        designs[lab] = dict((face, []) for face in faces)
    timestep_labels = dict.fromkeys(design_labels)
    for lab in design_labels:
        timestep_labels[lab] = dict((face, []) for face in faces)

    # Sort the files
    for f in files:
        if "Fiducial" in f:
            for lab in fiducial_labels:
                if not "Fiducial"+str(lab)+"_" in f:
                    continue
                for face in faces:
                    if "_0"+str(face)+"_" in f:
                        if append_prefix:
                            fiducials[lab][face].append(folder+f)
                        else:
                            fiducials[lab][face].append(f)

        elif "Design" in f:
            for lab in design_labels:
                if not "Design"+str(lab)+"_" in f:
                    continue
                for face in faces:
                    if "_0"+str(face)+"_" in f:
                        if append_prefix:
                            designs[lab][face].append(folder+f)
                        else:
                            designs[lab][face].append(f)

        else:
            print "Could not find a category for " + f

    # Sort and keep only the specified timesteps
    _timestep_sort(fiducials, timesteps)
    _timestep_sort(designs, timesteps, labels=timestep_labels)

    return fiducials, designs, timestep_labels


def _timestep_sort(d, timesteps, labels=None):
    '''
    Helper function for segmenting by timesteps.
    '''
    for lab in d.keys():
        for face in d[lab].keys():
            # Check for empty lists.
            if d[lab][face] == []:
                continue
            d[lab][face].sort()
            if timesteps == 'last':  # Grab the last one
                if labels is not None:
                    labels[lab][face].append(d[lab][face][-1][-16:-14])
                d[lab][face] = d[lab][face][-1]
            elif timesteps == 'max':  # Keep all available
                # Reverse the order so the comparisons are between the highest
                # time steps.
                d[lab][face] = d[lab][face][::-1]
            elif isinstance(timesteps, int):  # Slice out a certain section
                d[lab][face] = d[lab][face][:timesteps]
                if labels is None:
                    continue
                for val in d[lab][face]:
                    labels[lab][face].append(val[-16:-14])
            else:  # Make a copy and loop through the steps
                good_files = copy.copy(d[lab][face])
                for f in d[lab][face]:
                    match = ["_00"+str(step)+"_" in f for step in timesteps]
                    if not any(match):
                        good_files.remove(f)
                    if labels is not None:
                        labels[lab][face].append(f[-16:-14])
                d[lab][face] = good_files


def load_and_reduce(filename):
    '''
    Load the cube in and derive the property arrays.
    '''

    cube = getdata(filename, header=True)

    reduction = property_arrays(cube, rms_noise=0.001)
    reduction.return_all(save=False)

    return reduction.dataset

if __name__ == "__main__":

    # Call as:
    # python output.py path/to/folder/ 0 0 1 max fiducial0 T 10
    # The args correspond to: directory, fiducial number, face,
    # comparison face, time steps to use, output file prefix,
    # use multiple cores?, how many cores?

    from multiprocessing import Pool
    from itertools import izip, repeat

    statistics = ["Wavelet", "MVC", "PSpec", "Bispectrum", "DeltaVariance",
                  "Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF", "Cramer",
                  "Skewness", "Kurtosis", "VCS_Density", "VCS_Velocity",
                  "PDF"]  # "Dendrogram_Hist", "Dendrogram_Num"]

    print "Statistics to run: %s" % (statistics)
    num_statistics = len(statistics)

    # Read in cmd line args
    try:
        fiducial_num = int(sys.argv[2])
    except ValueError:
        fiducial_num = str(sys.argv[2])
    face = int(sys.argv[3])
    comp_face = int(sys.argv[4])
    try:
        timesteps = int(sys.argv[5])
    except ValueError:
        timesteps = str(sys.argv[5])
    save_name = str(sys.argv[6])
    MULTICORE = str(sys.argv[7])
    if MULTICORE == "T":
        MULTICORE = True
    else:
        MULTICORE = False
        NCORES = 1  # Placeholder to pass into run_all
    if MULTICORE:
        NCORES = int(sys.argv[8])

    # Read in all files in the given directory
    PREFIX = str(sys.argv[1])

    fiducials, designs, timesteps_labels = \
        files_sorter(PREFIX, timesteps="max",
                     append_prefix=True)

    if fiducial_num == "fid_comp":  # Run all the comparisons of fiducials

        print "Fiducials to compare %s" % (fiducials.keys())
        fiducial_index = []
        fiducial_col = []

        # number of comparisons b/w all fiducials
        num_comp = (len(fiducials)**2. - len(fiducials))/2
        # Change dim 2 to match number of time steps
        distances_storage = np.zeros((num_statistics, num_comp, 10))
        posn = 0
        prev = 0
        # no need to loop over the last one
        for fid_num, i in zip(fiducials.keys()[:-1],
                              np.arange(len(fiducials), 0, -1)):
            posn += i
            comparisons = fiducials.copy()
            for key in range(fid_num):
                del comparisons[key]
            partial_distances = \
                run_all(fiducials[fid_num], comparisons,
                        face, statistics, save_name, multicore=MULTICORE,
                        ncores=NCORES, comp_face=comp_face,
                        multi_timesteps=True, verbose=True)
            distances_storage[:, prev:posn, :] = partial_distances
            prev += i

            fiducial_index.extend(fiducials.keys()[fid_num:])

            fiducial_col.extend([posn-prev] * len(fiducials.keys()[fid_num:]))

        # consistent naming with non-fiducial case
        simulation_runs = fiducial_index
        face = comp_face
    else:  # Normal case of comparing to single fiducial

        distances_storage = \
            run_all(fiducials[fiducial_num], designs,
                    face, statistics, save_name,
                    multicore=MULTICORE, ncores=NCORES, comp_face=comp_face,
                    multi_timesteps=True)

        simulation_runs = designs.keys()
        fiducial_index = [fiducial_num] * len(designs.keys())

    # If using timesteps 'max', some comparisons will remain zero
    # To distinguish a bit better, set the non-comparisons to zero
    distances_storage[np.where(distances_storage == 0)] = np.NaN

    filename = save_name + str(face) + "_" + str(comp_face) + \
        "_distance_results.h5"
    from pandas import DataFrame, HDFStore, concat, Series

    ## Save data for each statistic in a dataframe.
    ## Each dataframe is saved in a single hdf5 file

    store = HDFStore("lustre/home/ekoch/results/"+filename)

    for i in range(num_statistics):
        # If timesteps is 'max', there will be different number of labels
        # in this case, don't bother specifying column names.
        if not 'max' in timesteps:
            df = DataFrame(distances_storage[i, :, :], index=simulation_runs,
                           columns=timesteps_labels[0][face])
        else:
            df = DataFrame(distances_storage[i, :, :], index=simulation_runs)

        if not "Fiducial" in df.columns:
            df["Fiducial"] = Series(fiducial_index, index=df.index)
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
