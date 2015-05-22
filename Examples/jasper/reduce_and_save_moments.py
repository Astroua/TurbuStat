
import numpy as np
from spectral_cube import SpectralCube, LazyMask
from astropy.io.fits import getdata
from astropy.wcs import WCS

from turbustat.data_reduction import Mask_and_Moments
from MPI import MPIPool

'''
Calculate the moments for all of the cubes.
'''


def reduce_and_save(filename, add_noise=False, rms_noise=0.001,
                    output_path="", cube_output=None,
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

    reduc = Mask_and_Moments(sc, scale=rms_noise)
    reduc.make_mask(mask=reduc.cube > nsig * reduc.scale)

    reduc.make_moments()
    reduc.make_moment_errors()

    # Remove .fits from filename
    save_name = filename.split("/")[-1][:-4]

    reduc.to_fits(output_path+save_name)

    # Save the noisy cube too
    if add_noise:
        if cube_output is None:
            reduc.cube.hdu.writeto(output_path+save_name)
        else:
            reduc.cube.hdu.writeto(cube_output+save_name)


def single_input(a):
    return reduce_and_save(*a)

if __name__ == "__main__":

    import sys
    import glob
    from itertools import repeat

    folder = str(sys.argv[1])

    output_folder = str(sys.argv[2])

    is_obs = str(sys.argv[3])
    if is_obs == "T":
        is_obs = True
    else:
        is_obs = False

    add_noise = str(sys.argv[4])
    if add_noise == "T":
        add_noise = True
    else:
        add_noise = False

    if add_noise:
        try:
            cube_output = str(sys.argv[5])
        except IndexError:
            print "Using same output folder for dirty cubes and moments."
            cube_output = output_folder
    else:
        cube_output = output_folder

    # Grab all of the fits files
    fits_files = glob.glob(folder+"*.fits")

    # Trying noise levels scaled by their brightness distribs
    if add_noise:
        rms_noise = 'scaled'
    elif is_obs:
        rms_noise = None
    else:
        rms_noise = 0.001

    np.random.seed(248954785)

    pool = MPIPool(loadbalance=False)

    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)

    pool.map(single_input, zip(fits_files,
                               repeat(add_noise),
                               repeat(rms_noise),
                               repeat(output_folder),
                               repeat(cube_output)))

    pool.close()
