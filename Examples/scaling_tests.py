
'''
Generate mock images/cubes and track the computational time + memory
requirements with increasing size.
'''

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Gaussian1D, Gaussian2D
from memory_profiler import memory_usage
# import tracemalloc
# import time
import pandas as pd

from turbustat.io.sim_tools import create_cube_header, create_image_header
from turbustat.simulator import make_extended

from turbustat.statistics import (Cramer_Distance, DeltaVariance,
                                  Dendrogram_Stats, Genus, MVC, PCA, PDF,
                                  PowerSpectrum, Bispectrum, SCF, StatMoments,
                                  Tsallis, VCA, VCS, Wavelet)


twod_stats = [DeltaVariance, Genus, MVC, PDF, PowerSpectrum,
              Bispectrum, StatMoments, Tsallis, Wavelet, Dendrogram_Stats]
threed_stats = [Cramer_Distance, PCA, SCF, VCA, VCS]

# WCS info for generated images/cubes
# These won't affect the timing, but it will create an HDU
# to mimic the default usage of most of the stats
pixel_scale = 1 * u.arcsec
beamfwhm = 1 * u.arcsec
restfreq = 1.42 * u.GHz
bunit = u.K

image_sizes = [32, 64, 128, 256, 512, 1024, 2048]
samp_time = [1e-4, 1e-3, 1e-2, 1e-2, 1e-2, 1e-1, 1e-1]

twod_results = {}
for Stat in twod_stats:
    twod_results[Stat.__name__ + "_memory"] = []
    twod_results[Stat.__name__ + "_time"] = []

threed_results = {}
for Stat in threed_stats:
    threed_results[Stat.__name__ + "_memory"] = []
    threed_results[Stat.__name__ + "_time"] = []

for size, del_t in zip(image_sizes, samp_time):

    img = np.abs(make_extended(size))
    img_hdr = create_image_header(pixel_scale, beamfwhm, (size, size),
                                  restfreq, bunit)

    img_hdu = fits.PrimaryHDU(img, img_hdr)

    # Run 2D stats
    for Stat in twod_stats:

        kwargs = {}
        run_kwargs = {}

        # MVC has different inputs than the rest
        if Stat == MVC:
            inputs = [img_hdu] * 3
        elif Stat == Wavelet:
            # Match the default number of scales to 25, same as Delta-Variance
            # Should eventually change default in code too
            inputs = [img_hdu]
            kwargs = {"num": 25}
            run_kwargs = {'convolve_kwargs': {'allow_huge': True}}
        elif Stat == DeltaVariance:
            inputs = [img_hdu]
            run_kwargs = {'allow_huge': True}
        else:
            inputs = [img_hdu]

        def stat_runner():
            out = Stat(*inputs, **kwargs).run(**run_kwargs)
            del out

        # From https://stackoverflow.com/questions/9850995/tracking-maximum-memory-usage-by-a-python-function/10117657#10117657
        usage = np.array(memory_usage(stat_runner, timestamps=True,
                                      interval=del_t))

        # First column is memory usage. Second is time.
        # Since the stat gets deleted, check if initial usage equals final
        # assert usage[0, 0] == usage[-1, 0]

        max_usage = usage.max(0)
        min_usage = usage.min(0)

        # In MB
        delta_mem = max_usage[0] - usage[0, 0]
        # In sec.
        delta_time = max_usage[1] - min_usage[1]

        # Using tracemalloc and time
        # tracemalloc.start()

        # one = tracemalloc.take_snapshot()
        # t1 = time.time()
        # out = Stat(*inputs, **kwargs).run()
        # t2 = time.time()
        # two = tracemalloc.take_snapshot()
        # del out
        # three = tracemalloc.take_snapshot()

        # tracemalloc.stop()

        # diff = two.compare_to(one, 'lineno')
        # # Looking for changes due to turbustat
        # delta_mem = 0
        # for dd in diff:
        #     if "turbustat" in "".join(dd.traceback.format()):
        #         delta_mem += dd.size_diff

        # delta_mem = delta_mem << u.B
        # delta_mem = delta_mem.to(u.MB).value

        # delta_time = t2 - t1

        twod_results[Stat.__name__ + "_memory"].append(delta_mem)
        twod_results[Stat.__name__ + "_time"].append(delta_time)

    # Now make a fake cube and run the cube stats.

    spec_pixel_scale = 1 * u.km / u.s

    cube_hdr = create_cube_header(pixel_scale, spec_pixel_scale,
                                  beamfwhm, (size, size, size),
                                  restfreq, bunit)

    # Build a cube from Gaussian models
    amps = Gaussian2D(1., size // 2, size // 2, size / 4., size / 4.)
    spec_model = Gaussian1D(1., 0., size / 4.)

    spec_axis = np.linspace(- size // 2, size // 2, size)
    spat_pos = np.arange(size)
    yy, xx = np.meshgrid(spat_pos, spat_pos)

    cube = np.zeros((size, size, size))

    for y, x in zip(yy.ravel(), xx.ravel()):
        cube[:, y, x] = amps(y, x) * spec_model(spec_axis)

    cube_hdu = fits.PrimaryHDU(cube, cube_hdr)

    # Run 3D stats
    for Stat in threed_stats:

        if Stat == Cramer_Distance:
            # Cramer is only a distance_metric
            def stat_runner():
                out = Stat(cube_hdu, cube_hdu).distance_metric()
                del out
        elif Stat == PCA:
            # Need to specify the min_eigval. Use prop of variance
            # PCA won't work on tiny cubes. Only do decomposition in that case
            def stat_runner():
                out = Stat(cube_hdu).run(min_eigval=0.995,
                                         eigen_cut_method='value',
                                         decomp_only=True)
                del out
        else:
            def stat_runner():
                out = Stat(cube_hdu).run()
                del out

        usage = np.array(memory_usage(stat_runner, timestamps=True, interval=0.01))

        # First column is memory usage. Second is time.
        # Since the stat gets deleted, check if initial usage equals final
        # assert usage[0, 0] == usage[-1, 0]

        max_usage = usage.max(0)
        min_usage = usage.min(0)

        # In MB
        delta_mem = max_usage[0] - usage[0, 0]
        # In sec.
        delta_time = max_usage[1] - min_usage[1]

        # Using tracemalloc and time
        # tracemalloc.start()

        # one = tracemalloc.take_snapshot()
        # t1 = time.time()
        # stat_runner()
        # t2 = time.time()
        # two = tracemalloc.take_snapshot()

        # tracemalloc.stop()

        # diff = two.compare_to(one, 'lineno')
        # # Looking for changes due to turbustat
        # # delta_mem = 0
        # # for dd in diff:
        # #     if "turbustat" in "".join(dd.traceback.format()):
        # #         delta_mem += dd.size_diff

        # delta_mem = sum([dd.size_diff for dd in diff])

        # delta_mem = delta_mem << u.B
        # delta_mem = delta_mem.to(u.MB).value

        # delta_time = t2 - t1

        threed_results[Stat.__name__ + "_memory"].append(delta_mem)
        threed_results[Stat.__name__ + "_time"].append(delta_time)

    del img, img_hdu, cube_hdu

# Save the outputs as csv files

twod_df = pd.DataFrame(twod_results, index=image_sizes)
twod_df.to_csv("twoD_scaling_tests.csv")

threed_df = pd.DataFrame(threed_results, index=image_sizes)
threed_df.to_csv("threeD_scaling_tests.csv")
