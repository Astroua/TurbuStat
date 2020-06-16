# Licensed under an MIT open source license - see LICENSE


'''
Create testing data sets from 2 cubes.
'''

import pytest


@pytest.mark.skip(reason='Only generates dataset1 and dataset2.npz.')
def make_datasets_npz():

    import numpy as np
    from astropy.io import fits

    # Set 1
    cube1 = fits.getdata(("/srv/astro/erickoch/enzo_sims/frac_factorial_set/"
        "Fiducial128_1.0.0/Fiducial128_1_21_0_0_flatrho_0021_13co/"
        "Fiducial128_1_21_0_0_flatrho_0021_13co.fits"))

    # Slice to a smaller size.
    cube1 = cube1[:, 25:57, 46:78]

    channel_max1 = np.nanmax(np.nanmax(cube1, axis=2), axis=1)

    keep_channel1 = np.where(channel_max1 > 0.001)[0]
    # We want to save which channels get kept in the output
    channel_tracker1 = np.zeros((2, cube1.shape[0]))
    channel_tracker1[0, :] = np.arange(cube1.shape[0])

    cond_cube1 = np.empty((len(keep_channel1), cube1.shape[1], cube1.shape[2]))

    for i, channel in enumerate(keep_channel1):
        cond_cube1[i, :, :] = cube1[channel, :, :]
        channel_tracker1[1, channel] = 1

    np.savez_compressed("dataset1", cube=cond_cube1,
                        channels=channel_tracker1)

    # Set 2
    cube2 = fits.getdata(("/srv/astro/erickoch/enzo_sims/frac_factorial_set/"
        "Design4.0.0/Design4_21_0_0_flatrho_0021_13co/"
        "Design4_21_0_0_flatrho_0021_13co.fits"))

    cube2 = cube2[:, 60:92, 50:82]

    channel_max2 = np.nanmax(np.nanmax(cube2, axis=2), axis=1)

    keep_channel2 = np.where(channel_max2 > 0.001)[0]
    channel_tracker2 = np.zeros((2, cube2.shape[0]))
    channel_tracker2[0, :] = np.arange(cube2.shape[0])

    cond_cube2 = np.empty((len(keep_channel2), cube2.shape[1], cube2.shape[2]))

    for i, channel in enumerate(keep_channel2):
        cond_cube2[i, :, :] = cube2[channel, :, :]
        channel_tracker2[1, channel] = 1

    np.savez_compressed("dataset2", cube=cond_cube2,
                        channels=channel_tracker2)
