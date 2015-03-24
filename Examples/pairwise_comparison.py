
'''
Compute pairwise comparisons between the timesteps of a sim.
'''

from turbustat.statistics import stats_wrapper
from turbustat.data_reduction import Mask_and_Moments

from astropy.io.fits import getdata
from astropy.wcs import WCS
from spectral_cube import SpectralCube, LazyMask
import numpy as np
from interruptible_pool import InterruptiblePool as Pool
from itertools import izip, combinations, repeat
from pandas import DataFrame


def pairwise(files, statistics=None, ncores=1, save=False,
             save_name='pairwise', add_noise=False,
             rms_noise=0.001):
    '''
    Create a distance matrix for a set of simulations.
    '''

    num = len(files)

    pos = np.arange(num)

    pool = Pool(processes=ncores)

    output = pool.map(single_input,
                      izip(repeat(files),
                           combinations(pos, 2),
                           repeat(statistics),
                           repeat(add_noise),
                           repeat(rms_noise)))

    pool.close()

    dist_matrices = np.zeros((len(output[0][0]), num, num))

    for out in output:
        for i, stat in enumerate(out[0].keys()):
            dist_matrices[i, out[1], out[2]] = out[0][stat]

    if save:
        stats = out[0].keys()

        for i, stat in enumerate(stats):
            df = DataFrame(dist_matrices[i, :, :])
            df.to_csv(save_name+'_'+stat+'_distmat.csv')
    else:
        return dist_matrices


def timestep_wrapper(files, pos, statistics, noise=False, rms_noise=0.001):

    pos1, pos2 = pos
    # Derive the property arrays assuming uniform noise (for sims)
    dataset1 = load_and_reduce(files[pos1], add_noise=noise,
                               rms_noise=rms_noise)
    dataset2 = load_and_reduce(files[pos2], add_noise=noise,
                               rms_noise=rms_noise)

    distances = stats_wrapper(dataset1, dataset2,
                              statistics=statistics, multicore=True)
    return distances, pos1, pos2


def single_input(a):
    return timestep_wrapper(*a)


def load_and_reduce(filename, add_noise=False, rms_noise=0.001,
                    nsig=3):
    '''
    Load the cube in and derive the property arrays.
    '''

    if add_noise:
        if rms_noise is None:
            raise TypeError("Must specify value of rms noise.")

        cube, hdr = getdata(filename, header=True)

        from scipy.stats import norm
        cube += norm.rvs(0.0, rms_noise, cube.shape)

        sc = SpectralCube(data=cube, wcs=WCS(hdr))

        mask = LazyMask(np.isfinite, sc)
        sc = sc.with_mask(mask)

    else:
        sc = filename

    reduc = Mask_and_Moments(sc, scale=rms_noise)
    reduc.make_mask(mask=reduc.cube > nsig * reduc.scale)
    reduc.make_moments()
    reduc.make_moment_errors()

    return reduc.to_dict()
