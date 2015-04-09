
'''
Pairwise comparisons between all sims at one timestep. Creates a distance
that may be compared to the other timesteps to look for clear trends in the
distance metric.
'''

from turbustat.statistics import stats_wrapper
from turbustat.data_reduction import Mask_and_Moments

from astropy.io.fits import getdata
from astropy.wcs import WCS
from spectral_cube import SpectralCube, LazyMask
import numpy as np
# from interruptible_pool import InterruptiblePool as Pool
from MPI import MPIPool
from itertools import izip, combinations, repeat
from pandas import DataFrame
from datetime import datetime


def pairwise(file_dict, pool, statistics=None, save=False,
             save_name='pairwise', save_path=None,
             add_noise=False, rms_noise=0.001):
    '''
    Create a distance matrix for a set of simulations.
    '''

    num = len(file_dict.keys())

    pos = file_dict.keys()

    output = pool.map(single_input,
                      zip(repeat(file_dict),
                          combinations(pos, 2),
                          repeat(statistics),
                          repeat(add_noise),
                          repeat(rms_noise)))

    i = 0
    while True:
        if output[i][0] == None:
            i += 1
        else:
            break

    dist_matrices = np.zeros((len(output[i][0]), num, num))

    for out in output:
        pos1 = pos.index(out[1])
        pos2 = pos.index(out[2])
        if out[0] == None:
            nans = [np.NaN] * dist_matrices.shape[0]

            dist_matrices[:, pos1, pos2] = nans
        else:

            for i, stat in enumerate(out[0].keys()):
                dist_matrices[i, pos1, pos2] = out[0][stat]

    if save:
        stats = out[0].keys()

        if save_path is None:
            save_path = ""

        for i, stat in enumerate(stats):
            df = DataFrame(dist_matrices[i, :, :], index=pos, columns=pos)
            df.to_csv(save_path+save_name+'_'+stat+'_distmat.csv')
    else:
        return dist_matrices


def timestep_wrapper(files_list, pos, statistics, noise=False,
                     rms_noise=0.001):

    pos1, pos2 = pos

    if files_list[pos1] == None or files_list[pos2] == None:
        return None, pos1, pos2

    print "On "+str(datetime.now())+" running %s %s" % (pos1, pos2)
    print "Files:  %s  %s" % (files_list[pos1], files_list[pos2])

    # Derive the property arrays assuming uniform noise (for sims)
    dataset1 = load_and_reduce(files_list[pos1], add_noise=noise,
                               rms_noise=rms_noise)
    dataset2 = load_and_reduce(files_list[pos2], add_noise=noise,
                               rms_noise=rms_noise)

    distances = stats_wrapper(dataset1, dataset2,
                              statistics=statistics, multicore=True)
    return distances, pos1, pos2


def single_input(a):
    return timestep_wrapper(*a)


def load_and_reduce(filename, add_noise=False, rms_noise=0.001,
                    nsig=3, no_moments=True):
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

    if not no_moments:

        reduc.make_moments()
        reduc.make_moment_errors()

        return reduc.to_dict()

    else:
        data_dict = {'cube': [reduc.cube.filled_data[:].value, reduc.cube.header]}

        return data_dict


if __name__ == "__main__":

    import sys
    import glob

    folder = str(sys.argv[1])
    ncores = int(sys.argv[2])
    tstep = str(sys.argv[3])
    face = str(sys.argv[4])
    output_folder = str(sys.argv[5])

    # Grab all of the fits files, then sort them by fiducial, design,
    # then by number, then by the face

    fits_files = glob.glob(folder+"*.fits")

    # Search for all files that have the timestep in the name
    fits_tstep = []

    for fits in fits_files:
        if "00"+tstep in fits and "_0"+face+"_" in fits:
            fits_tstep.append(fits)

    # put into dictionary so we can accurately track which Fid/Des it's from

    des = ["D"+str(i) for i in range(32)]
    fid = ["F"+str(i) for i in range(5)]

    des_fid = des + fid

    tstep_dict = dict.fromkeys(des_fid)

    for key in tstep_dict.keys():
        sim_type = 'Design' if key[0] == 'D' else 'Fiducial'
        num = key[1:]

        for fits in fits_tstep:
            if sim_type+num+"_" in fits:
                tstep_dict[key] = fits
                break

    pool = MPIPool(loadbalance=False)
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # pool = Pool(processes=ncores)

    pairwise(tstep_dict, pool, statistics=['Cramer', 'PCA'],
             save=True, save_name='SimSuite8_'+str(tstep)+"_"+str(face),
             save_path=output_folder)

    pool.close()
