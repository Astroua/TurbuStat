
import numpy as np
from astropy.io.fits import getdata
from astropy.wcs import WCS
from pandas import DataFrame
from itertools import combinations, izip, repeat
from datetime import datetime
import subprocess
try:
    from interruptible_pool import InterruptiblePool as Pool
except ImportError:
    from multiprocessing import Pool

from MPI import MPIPool

from turbustat.statistics import stats_wrapper, statistics_list
from turbustat.data_reduction import Mask_and_Moments
from spectral_cube import SpectralCube, LazyMask

'''
COMPLETE comparisons.

Inter-compare the observational cubes, as well as running against
the Fiducial runs
'''


def obs_to_obs(file_list, statistics, pool=None):
    '''
    Pass a list of the observational cubes.
    Reduce the data, run the statistics and output
    a csv table of the comparisons.

    If a pool is passed, it runs in parallel.
    '''

    num_comp = len(file_list) * (len(file_list) - 1) / 2

    distances = \
        DataFrame([(i, j) for i, j in
                  combinations(file_list, 2)],
                  columns=['Fiducial1', 'Fiducial2'])

    dendro_saves = \
        [(i[:-5]+"_dendrostat.pkl",
          j[:-5]+"_dendrostat.pkl")
         for i, j in combinations(file_list, 2)]

    for stat in statistics:
        distances[stat] = np.zeros((num_comp, ))

    generator = zip(combinations(file_list, 2),
                    repeat(statistics),
                    repeat(True),
                    dendro_saves)

    if pool is None:

        for i, combo in enumerate(generator):

            distance_dict = run_comparison(*combo)[0]

            for key in distance_dict.keys():
                distances[key][i] = distance_dict[key]

    else:

        outputs = pool.map(single_input, generator)

        for i, output in enumerate(outputs):

            distance_dict = output[0]

            for key in distance_dict.keys():
                distances[key][i] = distance_dict[key]

    return distances


def obs_to_fid(obs_list, fiducial_dict, statistics, pool=None):
    '''
    Treat observations as the designs.
    '''

    # Make the output
    output_size = (len(obs_list),
                   len(fiducial_dict[fiducial_dict.keys()[0]]))

    distances = {}

    for stat in statistics:
        distances[stat] = \
            np.zeros((len(obs_list),
                      len(fiducial_dict.keys())))

    for posn, obs in enumerate(obs_list):

        # Give dendrogram save file.
        dendro_saves = [None, obs[:-5]+"_dendrostat.pkl"]

        # Create generator
        gen = zip(zip(fiducial_dict.values(), repeat(obs)),
                  repeat(statistics),
                  repeat(True),
                  repeat(dendro_saves))


        print "On "+str(posn+1)+"/"+str(len(obs_list))+" at "+str(datetime.now())

        if pool is not None:
            outputs = pool.map(single_input, gen)
        else:
            outputs = map(single_input, gen)

        for output in outputs:

            pos1 = obs_list.index(output[2])
            pos2 = fiducial_dict.values().index(output[1])

            distance_dict = output[0]

            # Loop through statistics
            for key in distance_dict.keys():
                distances[key][pos1, pos2] = distance_dict[key]

    return distances


def des_to_obs(obs_list, design_dict, statistics, pool=None):
    '''
    Treat observations as the fiducials.
    '''

    # Make the output
    output_size = (len(design_dict[design_dict.keys()[0]]),
                   len(obs_list))

    distances = {}

    for stat in statistics:
        distances[stat] = \
            np.zeros((len(design_dict.keys()), len(obs_list)))

    for posn, obs in enumerate(obs_list):

        # Give dendrogram save file.
        dendro_saves = [obs[:-5]+"_dendrostat.pkl", None]

        # Create generator
        gen = zip(zip(repeat(obs), design_dict.values()),
                  repeat(statistics),
                  repeat(True),
                  repeat(dendro_saves))

        print "On "+str(posn+1)+"/"+str(len(obs_list))+" at "+str(datetime.now())

        if pool is not None:
            outputs = pool.map(single_input, gen)
        else:
            outputs = map(single_input, gen)

        for output in outputs:

            pos1 = design_dict.values().index(output[2])
            pos2 = obs_list.index(output[1])

            distance_dict = output[0]

            # Loop through statistics
            for key in distance_dict.keys():
                distances[key][pos1, pos2] = distance_dict[key]

    return distances


def run_comparison(fits, statistics, add_noise, dendro_saves=[None, None]):

    fits1, fits2 = fits

    # Derive the property arrays assuming uniform noise (for sims)
    fiducial_dataset = load_and_reduce(fits1)
    testing_dataset = load_and_reduce(fits2)

    if add_noise:
        vca_break = 1.5
        vcs_break = -0.5
    else:
        vca_break = None
        vcs_break = -0.8

    distances = stats_wrapper(fiducial_dataset, testing_dataset,
                              statistics=statistics, multicore=True,
                              vca_break=vca_break, vcs_break=vcs_break,
                              dendro_saves=dendro_saves)

    return distances, fits1, fits2


def single_input(a):
    return run_comparison(*a)


def load_and_reduce(filename, moment_folder="moments/"):
    '''
    Load the cube in and derive the property arrays.
    '''

    file_dict = {}

    file_labels = ["_moment0", "_centroid", "_linewidth", "_intint"]
    labels = ["moment0", "centroid", "linewidth", "integrated_intensity"]

    # load the cube in
    file_dict['cube'] = list(getdata(filename, header=True))

    prefix_direc = "/".join(filename.split("/")[:-1])
    if len(prefix_direc) != 0:
        prefix_direc = prefix_direc + "/"
    sim_name = filename.split("/")[-1][:-4]

    for dic_lab, file_lab in zip(labels, file_labels):
        file_dict[dic_lab] = \
            list(getdata(prefix_direc+moment_folder+sim_name+file_lab+".fits", 0, header=True))

        # And the errors
        file_dict[dic_lab+"_error"] = \
            list(getdata(prefix_direc+moment_folder+sim_name+file_lab+".fits", 1, header=True))

    return file_dict


def sort_distances(statistics, distances):
    if len(statistics) > 1:
        distance_array = np.empty((len(distances), len(statistics)))
    elif len(statistics) == 1:
        distance_array = np.empty((len(distances), 1))

    for j, dist in enumerate(distances):
        distance_array[j, :] = [dist[stat] for stat in statistics]

    return distance_array


def sort_sim_files(sim_list, sim_labels=np.arange(0, 5),
                   timestep_labels=np.arange(21, 31),
                   sim_type='Fiducial'):
    '''
    Sort by the given labels.
    '''

    sim_dict = dict.fromkeys(sim_labels)

    for key in sim_dict.keys():
        sim_dict[key] = dict.fromkeys(timestep_labels)

    for sim in sim_list:
        for label in sim_labels:
            if sim_type+str(label)+"_" in sim:
                key = label
                break
        else:
            raise TypeError("Cannot find appropriate label for: "+sim)

        for time in timestep_labels:
            if "_00"+str(time)+"_" in sim:
                tstep = time
                break
        else:
            raise TypeError("Cannot find appropriate timestep for: "+sim)

        # Remove empty timesteps
        sim_dict[key] =\
            dict((k, v) for k, v in sim_dict[key].iteritems() if v is not None)

        sim_dict[key][tstep] = sim

    return sim_dict


if __name__ == "__main__":

    import os
    import sys
    from pandas import DataFrame

    # statistics =  statistics_list

    # Set to run on the 'good' statistics
    statistics = ["DeltaVariance", "VCS", "VCS_Density", "VCS_Velocity",
                  "VCA", "PCA", "SCF", "Cramer", "VCS_Break", "Dendrogram_Hist",
                  "Dendrogram_Num"]

    print "Statistics to run: %s" % (statistics)


    obs_dir = sys.argv[1]
    sim_dir = sys.argv[2]
    face = sys.argv[3]

    # Type of comparison
    comparison = str(sys.argv[4])
    valid_comp = ["Obs_to_Fid", "Obs_to_Obs", "Des_to_Obs"]
    if comparison not in valid_comp:
        raise Warning("comparison type give is not valid: " + str(comparison))

    # Set parallel type
    multiproc = sys.argv[5]
    valid_proc = ["MPI", "noMPI", "Single"]
    if multiproc not in valid_proc:
        raise Warning("multiproc type give is not valid: " + str(multiproc))

    # Output results directory
    output_dir = sys.argv[6]

    if output_dir[-1] != "/":
        output_dir += "/"

    save_name = sys.argv[7]

    # Load the list of complete cubes in

    obs_cubes = [obs_dir+f for f in os.listdir(obs_dir) if f[-4:] == 'fits']

    # sim_dir = "/Volumes/RAIDers_of_the_lost_ark/SimSuite8/"

    # Toggle the pool on here

    if multiproc == "MPI":
        pool = MPIPool(loadbalance=False)

        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
    elif multiproc == "noMPI":
        pool = Pool(nprocesses=12)
    else:
        pool = None

    # Do the actual comparisons

    if comparison == "Obs_to_Fid":
        sim_cubes = [sim_dir+f for f in os.listdir(sim_dir) if "Fiducial" in f
                     and "_0"+face+"_" in f]

        sim_dict = sort_sim_files(sim_cubes)

        output_storage = []

        # Loop through the fiducial sets
        for fid in sim_dict.keys():

            distances = obs_to_fid(obs_cubes, sim_dict[fid], statistics,
                                   pool=pool)

            output_storage.append(distances)

        if pool is not None:
            pool.close()

        # Loop through the statistics and append to the HDF5 file.

        from pandas import HDFStore

        for fid, distances in enumerate(output_storage):

            store = HDFStore(output_dir+save_name+"_fiducial_"+str(fid)+
                             "_face_"+str(face)+".h5")

            for key in distances.keys():

                df = DataFrame(distances[key],
                               index=[obs.split("/")[-1]
                                      for obs in obs_cubes],
                               columns=[sim.split("/")[-1]
                                        for sim in sim_dict[fid].values()])

                store[key] = df

            store.close()

    elif comparison == "Des_to_Obs":
        sim_cubes = [sim_dir+f for f in os.listdir(sim_dir) if "Design" in f
                     and "_0"+face+"_" in f]

        sim_dict = sort_sim_files(sim_cubes, sim_labels=np.arange(0, 32),
                                  sim_type="Design")

        output_storage = []

        # Loop through the fiducial sets
        for des in sim_dict.keys():

            distances = des_to_obs(obs_cubes, sim_dict[des], statistics,
                                   pool=pool)

            output_storage.append(distances)

        if pool is not None:
            pool.close()

        # Loop through the statistics and append to the HDF5 file.

        from pandas import HDFStore

        for des, distances in enumerate(output_storage):

            store = HDFStore(output_dir+save_name+"_design_"+str(des)+
                             "_face_"+str(face)+".h5")

            for key in distances.keys():

                df = DataFrame(distances[key],
                               index=[sim.split("/")[-1]
                                      for sim in sim_dict[des].values()],
                               columns=[obs.split("/")[-1]
                                        for obs in obs_cubes])

                store[key] = df

            store.close()
    else:
        # Pairwise comparisons between the observations only

        complete_distances = obs_to_obs(obs_cubes, statistics, pool=pool)

        if pool is not None:
            pool.close()

        complete_distances.to_csv("complete_comparisons.csv")

        # for i, stat in enumerate(complete_distances.keys()):

        #     df = DataFrame(complete_distances[stat], index=obs_cubes,
        #                    columns=obs_cubes)

        #     df.to_csv(obs_dir+"complete_comparisons_"+stat+".csv")
