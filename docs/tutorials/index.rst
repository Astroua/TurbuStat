
*******************
TurbuStat Tutorials
*******************

Tutorials are provided for each of the statistic classes and their associated distance metric classes. The tutorials use the same two data sets, described on the :ref:`data for tutorials <data_for_tutorial>` page.

The plotting routines are highlighted in each of the tutorials. For users who require custom plotting routines, we recommend looking at the `plotting source code <https://github.com/Astroua/TurbuStat/tree/master/turbustat/statistics>`_ as a starting point.

.. toctree::
    :maxdepth: 2

    data_for_tutorials
    applying_apodizing_functions
    correcting_for_the_beam
    missing_data_noise

Statistics
==========

.. toctree::
    :maxdepth: 2

    statistics/running_statistics
    statistics/bispectrum_example
    statistics/delta_variance_example
    statistics/dendrogram_example
    statistics/genus_example
    statistics/mvc_example
    statistics/pca_example
    statistics/pdf_example
    statistics/pspec_example
    statistics/scf_example
    statistics/statmoments_example.rst
    statistics/tsallis_example.rst
    statistics/vca_example
    statistics/vcs_example
    statistics/wavelet_example

Distance Metrics
================

This section describes the distance metrics defined in `Koch et al. 2017 <https://ui.adsabs.harvard.edu/#abs/2017MNRAS.471.1506K/abstract7>`_ for comparing two data sets with some output of the statistics listed above. It is important to note that few of these distance metrics are defined to be absolute. Rather, most of the metrics give *relative* distances and are defined only when comparing with a common fiducial image.

As shown in `Koch et al. 2017 <https://ui.adsabs.harvard.edu/#abs/2017MNRAS.471.1506K/abstract7>`_, the distance metrics for some statistics have more scatter than others.  Some metrics also suffer from systematic issues and should be avoided when those systematics cannot be controlled for.  The **Cramer distance metric** is an example of this; its shortcomings are described in the paper linked above, and while the implementation is still available, we recommend caution when using it.

A distance metric for **Tsallis** statistics has not been explored and is not currently available in this release.

.. toctree::
    :maxdepth: 2

    metrics/running_metrics
    metrics/bispectrum_example
    metrics/cramer_example
    metrics/delvar_example
    metrics/dendro_example
    metrics/genus_example
    metrics/mvc_example
    metrics/pca_example
    metrics/pdf_example
    metrics/pspec_example
    metrics/scf_example
    metrics/statmoments_example
    metrics/vca_example
    metrics/vcs_example
    metrics/wavelet_example
