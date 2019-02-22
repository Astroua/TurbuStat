.. _runmetrics:


**********************
Using distance metrics
**********************

Distance metrics run a statistic on two datasets and use an output of the statistic to quantitatively determine how similar the datasets are. Like the statistic classes, the distance metrics are also `python classes <https://docs.python.org/3/tutorial/classes.html>`_.

.. warning:: Using the distance metrics requires some understanding of how the statistics are computed. Please read the :ref:`introduction to using statistics <runstats>` page.

There is less structure in the distance metric classes compared to the statistic classes. There are two steps to using the distance metrics:

1. **Initialization** -- Like the statistics, unchanging information is given here. The two datasets are given here. However, the statistics are also run from this step, so properties often given to the `run` command for the statistics should be given here.

.. note:: The distance metrics do not always use the full information from a statistic. In these cases, there will be fewer arguments to specify than the statistic's `run` function.

For the `~turbustat.statistics.Wavelet_Distance`, the distance metric is initialized using:

    >>> from turbustat.statistics import Wavelet_Distance
    >>> from astropy.io import fits
    >>> hdu = fits.open("file.fits")[0]  # doctest: +SKIP
    >>> hdu2 = fits.open("file2.fits")[0]  # doctest: +SKIP
    >>> wave_dist = Wavelet_Distance(hdu, hdu2, xlow=[5, 7] * u.pix,
    ...                              xhigh=[30, 35] * u.pix)  # doctest: +SKIP

The two datasets are given. For the wavelet transform, the distance is the absolute difference between the slopes, normalized by the square sum of the slope uncertainties. Thus it is important to set the scales that the transform is fit to. Different limits can be given for the two datasets, as shown above, by specifying `xlow` and `xhigh` with two different values. If only a single value is given, it will be used for both datasets. Most parameters that can be specified for one or both of the datasets use this format.

Alternatively, if the wavelet transform has already been computed for one or both of the datasets, the `~turbustat.statistics.Wavelet` can be passed instead of the dataset:

    >>> from turbustat.statistics import Wavelet, Wavelet_Distance
    >>> wave1 = Wavelet(hdu).run()  # doctest: +SKIP
    >>> wave2 = Wavelet(hdu).run()  # doctest: +SKIP
    >>> wave_dist = Wavelet_Distance(wave1, wave2)  # doctest: +SKIP

Some distance metrics require that the statistic be computed with a common set of bins or spatial scales. If a `~turbustat.statistics.Wavelet` is given that does not satisfy the criteria, it will be re-run with `~turbustat.statistics.Wavelet_Distance`. The criteria to avoid re-computing a statistic is specified as *Notes* in the distance metric docstrings. See the source code documentation on this site.

2. The second step is to compute the distance metric. In nearly all cases, computing the distance metric is much faster than computing the statistics. All of the distance metric classes have a `distance_metric` function:

    >>> wave_dist.distance_metric(verbose=True)  # doctest: +SKIP
    >>> wave_dist.distance  # doctest: +SKIP

The distance is usually an attribute called `distance`. Different names are used when there are multiple distance metrics defined. For example, the `~turbustat.statistics.PDF_Distance` has `hellinger_distance`, `ks_distance`, `lognormal_distance`. See the source code documentation for specifics on each distance metric class. When multiple distance metrics are defined, there are often multiple functions to compute each metric. The `distance_metric` function will usually run all of the distance metrics.

With `verbose=True` in `distance_metric`, a summary plot showing both datasets will be returned. Usually these plots call the plotting function from the statistic classes. Labels to show in the legend for the two datasets can be given.


.. note:: The one metric that differs from the rest is the `~turbustat.statistics.Cramer_Distance`. This metric does not have a statistic class because it is a two-sample statistical test. The structure is the same as the rest of the distance metrics but contains more steps to compute the method.
