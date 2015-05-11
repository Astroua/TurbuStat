
import numpy as np
from astropy.io.fits import getdata
from itertools import combinations, izip, repeat
try:
    from MPI import InterruptiblePool as Pool
except ImportError:
    from multiprocessing import Pool

from turbustat.statistics import stats_wrapper
from turbustat.data_reduction import Mask_and_Moments

'''
COMPLETE comparisons.

Inter-compare the observational cubes, as well as running against
the Fiducial runs
'''


def obs_to_obs(file_list, statistics, nsig=3, pool=None):
    '''
    Pass a list of the observational cubes.
    Reduce the data, run the statistics and output
    a csv table of the comparisons.

    If a pool is passed, it runs in parallel.
    '''



    distances = {}

    for stat in statistics:
     distances[stat] = np.zeros((len(file_list), len(file_list)))

    generator = izip(combinations(file_list, 2),
                     repeat(statistics),
                     repeat(False),
                     repeat(None),
                     repeat(nsig))

    if pool is None:

        for combo in generator:

            pos1 = file_list.index(combo[0][0])
            pos2 = file_list.index(combo[0][1])

            distance_dict = run_comparison(*combo)[0]

            for key in distance_dict.keys():
                distances[key][pos1, pos2] = distance_dict[key]

    else:

        outputs = pool.map(single_input, generator)

        for output in outputs:

            pos1 = file_list.index(output[1])
            pos2 = file_list.index(output[2])

            distance_dict = output[0]

            for key in distance_dict.keys():
                distances[key][pos1, pos2] = distance_dict[key]

    return distances


def obs_to_fid(obs_list, fiducial_dict, pool=None):
    pass


def run_comparison(fits, statistics, add_noise, rms_noise, nsig):

    fits1, fits2 = fits

    # Derive the property arrays assuming uniform noise (for sims)
    fiducial_dataset = load_and_reduce(fits1, add_noise=add_noise,
                                       rms_noise=rms_noise, nsig=nsig)
    testing_dataset = load_and_reduce(fits2, add_noise=add_noise,
                                      rms_noise=rms_noise, nsig=nsig)

    if add_noise:
        vca_break = 1.5
        vcs_break = -0.5
    else:
        vca_break = None
        vcs_break = -0.8

    distances = stats_wrapper(fiducial_dataset, testing_dataset,
                              statistics=statistics, multicore=True,
                              vca_break=vca_break, vcs_break=vcs_break)

    return distances, fits1, fits2


def single_input(a):
    return run_comparison(*a)


def load_and_reduce(filename, add_noise=False, rms_noise=None,
                    nsig=3, slicewise_noise=True):
    '''
    Load the cube in and derive the property arrays.
    '''

    if add_noise:
        if rms_noise is None:
            raise TypeError("Must specify value of rms noise.")

        cube, hdr = getdata(filename, header=True)

        # Optionally scale noise by 1/10th of the 98th percentile in the cube
        if rms_noise == 'scaled':
            rms_noise = 0.1*np.percentile(cube[np.isfinite(cube)], 98)

        from scipy.stats import norm
        if not slicewise_noise:
            cube += norm.rvs(0.0, rms_noise, cube.shape)
        else:
            spec_shape = cube.shape[0]
            slice_shape = cube.shape[1:]
            for i in range(spec_shape):
                cube[i, :, :] += norm.rvs(0.0, rms_noise, slice_shape)

        sc = SpectralCube(data=cube, wcs=WCS(hdr))

        mask = LazyMask(np.isfinite, sc)
        sc = sc.with_mask(mask)

    else:
        sc = filename

    if rms_noise is None:
        reduc = Mask_and_Moments(sc)
    else:
        reduc = Mask_and_Moments(sc, rms_noise=rms_noise)
    reduc.make_mask(mask=reduc.cube > nsig * reduc.scale)
    reduc.make_moments()
    reduc.make_moment_errors()

    return reduc.to_dict()

if __name__ == "__main__":

    import os
    from pandas import DataFrame

    statistics = ["Wavelet", "MVC", "PSpec", "Bispectrum", "DeltaVariance",
                  "Genus", "VCS", "VCA", "Tsallis", "PCA", "SCF", "Cramer",
                  "Skewness", "Kurtosis", "VCS_Density", "VCS_Velocity",
                  "VCS_Break", "PDF", "Dendrogram_Hist", "Dendrogram_Num"]

    print "Statistics to run: %s" % (statistics)


    # Load the list of complete cubes in

    complete_dir = "/Users/eric/Data/complete/"

    complete_cubes = [complete_dir+f for f in os.listdir(complete_dir) if f[-4:] == 'fits']

    # Run the complete comparisons

    complete_distances = obs_to_obs(complete_cubes, statistics)

    for i, stat in enumerate(complete_distances.keys()):

        df = DataFrame(complete_distances[stat], index=complete_cubes,
                       columns=complete_cubes)

        df.to_csv(complete_dir+"complete_comparisons_"+stat+".csv")


    # Load fiducial cubes in, then create a dictionary separating by face and
    # timestep.