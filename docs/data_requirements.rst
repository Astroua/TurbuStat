.. _data_reqs:


*****************
Data Requirements
*****************

Use of the statistics and distance metrics require the input data to satisfy some criteria.

Spatial Projection
******************

TurbuStat assumes that the spatial dimensions of the data **are square on the sky**. All physical and angular dimensions will be incorrect, otherwise.  Data with non-square pixels should first be reprojected. This can be easily done using `spectral-cube <http://spectral-cube.readthedocs.io/en/latest/>`_:

    >>> reproj_cube = cube.reproject(new_header)  # doctest: +SKIP
    >>> reproj_proj = proj_2D.reproject(new_header_2D)  # doctest: +SKIP

Considerations for distance metrics
***********************************

The correct pre-processing of data sets is crucial for attaining a meaningful comparison. Listed here are pre-processing considerations that should be considered when using the distance metrics.

The extent of these effects will differ for different data sets. We recommend testing a subset of the data by placing the data at a common resolution (physical or angular) and grid-size, depending on your application. Smoothing and reprojection operations are straightforward to perform with the `spectral-cube package <http://spectral-cube.readthedocs.io/en/latest/smoothing.html>`_.

1. **Spatial scales** -- Unlike many of the statistics, the distance metrics do not consider physical spatial units.  Specifically, metrics that directly compare values at specific scales (`~turbustat.statistics.DeltaVariance_Distance`, `~turbustat.statistics.SCF_Distance`) will use the given WCS information for the data sets to find a common angular scale.

    Different spatial sizes are less of a concern for slope-based comparisons, but discrepancies can arise if the data sets have different noise levels. In these cases, the scales used to fit to the power-spectrum (or the equivalent statistic output) should be chosen carefully to avoid bias from the noise. A similar issue can arise when comparing different simulated data sets if the simulations have different inertial ranges.


2. **Spectral scales** -- The spectral sampling and range should be considered for all methods that utilize the entire PPV cube (SCF, VCA, VCS, dendrograms, PCA, PDF). The issue with using different-sized spectral pixels affects the noise properties, and in some statistics, the measurement itself.

    For the former, the noise level can introduce a bias in the measured quantities.  To mitigate this, data can be masked prior to running metrics.  Otherwise, minimum cut-off values can be specified for metrics that utilize the actual intensity values of the data, such as dendrograms and the PDF.  For statistics that are independent of intensity, like a power-law slope or correlation, the fitting range can be specified for each statistic to minimize bias from noise. This is the same effect described above for spatial scales.

    For the second case, the VCA index is *expected* to change with spectral resolution depending on the underlying properties of the turbulent fields (see the :ref:`VCA tutorial <vca_tutorial>`).

Data units for distance metrics
*******************************

Most of the distance metrics will not depend on the absolute value of the data sets. The exceptions are when values of a statistic are directly compared. This includes `~turbustat.statistics.Cramer_Distance`, the curve distance in `~turbustat.statistics.DeltaVariance_Distance`, and the bins used in the histograms of `~turbustat.statistics.StatMoments_Distance` and `~turbustat.statistics.PDF_Distance`.  While each of these methods applies some normalization scheme to the data, we advise converting both data sets to a common unit to minimize possible discrepancies.
