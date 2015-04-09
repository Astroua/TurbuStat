
'''
Compute pairwise comparisons between the timesteps of a sim.
'''

from turbustat.statistics import stats_wrapper
from turbustat.data_reduction import Mask_and_Moments

from astropy.io.fits import getdata
from astropy.wcs import WCS
from spectral_cube import SpectralCube, LazyMask
import numpy as np
from ..interruptible_pool import InterruptiblePool as Pool
from itertools import izip, combinations, repeat
from pandas import DataFrame
from datetime import datetime


def pairwise(fid_files, des_files, statistics=None, ncores=1, save=False,
             save_name='pairwise', save_path=None,
             add_noise=False, rms_noise=0.001):
    '''
    Create a distance matrix for a set of simulations.
    '''

    num = min(len(fid_files), len(des_files))

    pos = np.arange(num)

    pool = Pool(processes=ncores)

    output = pool.map(single_input,
                      izip(repeat(fid_files),
                           repeat(des_files),
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

        if save_path is None:
            save_path = ""

        for i, stat in enumerate(stats):
            df = DataFrame(dist_matrices[i, :, :])
            df.to_csv(save_path+save_name+'_'+stat+'_distmat.csv')
    else:
        return dist_matrices


def timestep_wrapper(fid_files, des_files, pos, statistics, noise=False,
                     rms_noise=0.001):

    pos1, pos2 = pos
    # Derive the property arrays assuming uniform noise (for sims)
    dataset1 = load_and_reduce(fid_files[pos1], add_noise=noise,
                               rms_noise=rms_noise)
    dataset2 = load_and_reduce(des_files[pos2], add_noise=noise,
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


if __name__ == "__main__":

    import sys
    import glob

    folder = str(sys.argv[1])
    ncores = int(sys.argv[2])
    des_or_fid = str(sys.argv[3])
    start_num = int(sys.argv[4])
    end_num = int(sys.argv[5])

    output_folder = str(sys.argv[6])

    # Grab all of the fits files, then sort them by fiducial, design,
    # then by number, then by the face

    fits_files = glob.glob(folder+"*.fits")

    fiducials = []
    designs = []

    for fits in fits_files:
        if "Fiducial" in fits:
            fiducials.append(fits)
        elif "Design" in fits:
            designs.append(fits)
        else:
            print "Cannot classify %f" % (fits)

    # This set has 5 fiducials and 32 designs

    fid_num = np.arange(5)
    des_num = np.arange(start_num, end_num+1)

    faces = np.arange(3)

    fids_dict = {}
    des_dict = {}

    for num in fid_num:
        fids_dict[num] = dict.fromkeys(faces)
        for face in faces:
            fids_dict[num][face] = []

        for fid in fiducials:
            if not "Fiducial"+str(num) in fid:
                continue

            for face in faces:
                if "_0"+str(face)+"_" in fid:
                    fids_dict[num][face].append(fid)
                    break

    for num in des_num:
        des_dict[num] = dict.fromkeys(faces)
        for face in faces:
            des_dict[num][face] = []

        for des in designs:
            if not "Design"+str(num)+"_" in des:
                continue

            for face in faces:
                if "_0"+str(face)+"_" in des:
                    des_dict[num][face].append(des)
                    break

    # Now run the pairwise comparisons

    if des_or_fid == 'Fiducial':

        for num1 in fid_num:
            for num2 in fid_num:
                print "On Fiducial "+str(num)+" of "+str(max(fid_num))
                for face in faces:
                    print 'Face '+str(face)+" at "+str(datetime.now())
                    pairwise(fids_dict[num][face], ncores=ncores, save=True,
                             statistics=['Cramer', 'PCA', 'PDF', 'SCF',
                                         'VCA', 'VCS'],
                             save_name='SimSuite8_Fiducial'+str(num1)+"_"+str(num2)+"_face"+str(face),
                             save_path=output_folder)

    elif des_or_fid == 'Design':
        for num in des_num:
            print "On Design "+str(num)+" of "+str(max(des_num))
            for face in faces:
                print 'Face '+str(face)+" at "+str(datetime.now())
                pairwise(des_dict[num][face], ncores=ncores, save=True,
                         statistics=['Cramer', 'PCA', 'PDF', 'SCF',
                                     'VCA', 'VCS'],
                         save_name='SimSuite8_Design'+str(num)+"_"+str(face),
                         save_path=output_folder)

    else:
        raise TypeError("Must be 'Fiducial' or 'Design'.")
