.. _data_reqs:


*****************
Data Requirements
*****************

Use of the statistics and distance metrics require the input data to satisfy some criteria.

Spatial Projection
******************

TurbuStat assumes that the spatial dimensions of the data are square. All physical and angular dimensions will be incorrect, otherwise.  Data with non-square pixels should first be reprojected. This can be easily done using `spectral-cube <http://spectral-cube.readthedocs.io/en/latest/>`_::

    >>> reproj_cube = cube.reproject(new_header)
    >>> reproj_proj = proj_2D.reproject(new_header_2D)

Considerations for distance metrics
***********************************


The correct pre-processing of data sets is crucial for attaining a meaningful comparison. Listed here are pre-processing considerations that should be heeded when using the distance metrics.

The extent of these effects will differ for different data sets. We recommend testing a subset of the data by placing the data at a common resolution (physical or angular) and grid-size, depending on your application. The smoothing and reprojection operations straightforward to perform with the `spectral-cube package <http://spectral-cube.readthedocs.io/en/latest/smoothing.html>`_.

1. Spatial scales -- Unlike many of the statistics, the distance metrics do not consider physical spatial units. Instead they compare outputs based on the angular scales of the data.

For example, consider `~turbustat.statistics.DeltaVariance_Distance` with different pixel scales::

XXX finish example XXX

the lags of the delta-variance

The choice of lags for the `~turbustat.statistics.SCF_Distance` will work similarly.

Different spatial sizes are less of a concern for slope-based comparisons, but discrepancies can arise if the data sets have different noise levels. In these cases, the scales used to fit to the power-spectrum (or the equivalent statistic output) should be chosen carefully to avoid bias from the noise. A similar issue can arise when comparing different simulated data sets if the simulations have different inertial ranges.


2. Spectral scales -- The spectral sampling and range should be considered for all methods that utilize the entire PPV cube (SCF, VCA, VCS, dendrograms, PCA, PDF). The issue with using different-sized spectral pixels affects the noise properties, and in some statistics, the measurement itself.

For the former, the noise level can introduce a bias in the measured quantities.  To mitigate this, data can be masked prior to running metrics (XXX see mask and moment XXX).  Otherwise, minimum cut-off values can be specified for metrics the utilize the actual intensity values of the data, such as dendrograms and the PDF.  For statistics that are independent of intensity, such as a power-law slope or correlation, the fitting range can be specified for each statistic to minimize bias from noise. This is the same effect described above for spatial scales.

For the second case, the VCS and VCA properties are _expected_ to change with different spectral pixel sizes. For the VCS, the slopes and the transition point depend on the largest spatial scale and the beam size of the data (see XXX vcs tutorial XXX). The slope of the VCA will also vary with the spectral pixel size, which depends on the underlying properties of the turbulent fields (see XXX vca tutorial XXX).
