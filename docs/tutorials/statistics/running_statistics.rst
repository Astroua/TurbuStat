.. _runstats:


************************
Using statistics classes
************************

The statistics implemented in TurbuStat are `python classes <https://docs.python.org/3/tutorial/classes.html>`_. This structure allows for derived properties to persist without having to manually carry them forward through each step.

Using most of the statistic classes will involved two steps:

1. **Initialization** -- The data, relevant WCS information, and other unchanging properties (like the distance) are specified here. Some of the statistics calculated at specific scales (like `~turbustat.statistics.Wavelet` or `~turbustat.statistics.SCF`) can have those scales set here, too. Below is an example using `~turbustat.statistics.Wavelet`:

    >>> from turbustat.statistics import Wavelet
    >>> import numpy as np
    >>> from astropy.io import fits
    >>> from astropy.units import u
    >>> hdu = fits.open("file.fits")[0]  # doctest: +SKIP
    >>> spatial_scales = np.linspace(0.1, 1.0, 20) * u.pc   # doctest: +SKIP
    >>> wave = Wavelet(hdu, scales=spatial_scales,
    ...                distance=260 * u.pc)  # doctest: +SKIP

2. **The run function** -- For most use-cases, the `run` function can be used to compute the statistic. All of the statistics have this function. It will compute the statistic, perform any relevant fitting, and optionally create a summary plot. The docstring for each of the `run` functions describe the parameters that can be changed from this function. The parameters that are critical to the behaviour of the statistic can all be set from `run`. Continuing with the `~turbustat.statistics.Wavelet` example from above, the `run` function is called as:

    >>> wave.run(verbose=True, xlow=0.2 * u.pc, xhigh=0.8 * u.pc)  # doctest: +SKIP

This function will run the wavelet transform and fit the relation between the given bounds (`xlow` and `xhigh`). With `verbose=True`, a summary plot is returned.


What if you need to set parameters not accessible from `run`? `run` wraps multiple steps in one function, however, the statistic can be run in steps when fine-tuning is needed for a particular data set. Each of the statistics has at least one computational step. For `~turbustat.statistics.Wavelet`, there are two steps: (1) computing the transform (`~turbustat.statistics.Wavelet.compute_transform`) and (2) fitting the log of the transform (`~turbustat.statistics.Wavelet.fit_transform`). Running these two functions is equivalent to using `run`.

The statistics also have plotting functions. From `run`, these functions are called whenever `verbose=True` is given. All of the plotting functions start with `plot_`; for `~turbustat.statistics.Wavelet`, the plotting function is `~turbustat.statistics.Wavelet.plot_transform`. Supplying a `save_name` to this function will save the figure, the x-units can also be set for spatial transforms (like the wavelet transform) as pixel, angular, or physical (when a distance is given) units, and additional arguments can be given to set the colours and markers used in the plot.

Statistic classes can also be saved or loaded as pickle files. Saving is performed with the `save_results` function:

    >>> wave.save_results("wave_file.pkl", keep_data=False)  # doctest: +SKIP

Whether to include the data in saved file is set with `keep_data`. By default, the data is *not* saved to save storage space.

.. note:: If the statistic is not saved with the data, it cannot be recomputed after loading.

Loading the statistic from a saved file uses the `load_results` function:

    >>> new_wave = Wavelet.load_results("wav_file.pkl")  # doctest: +SKIP

Unless the data is saved, everything but the data is new accessible from `new_wave`.
