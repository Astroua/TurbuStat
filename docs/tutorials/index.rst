
*******************
TurbuStat Tutorials
*******************

.. toctree::
    :maxdepth: 2

    preprocessing_data
    masking_and_moments
    applying_apodizing_functions
    correcting_for_the_beam

Statistics
==========

.. toctree::
    :maxdepth: 2

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

These describe the distance metrics defined in `Koch et al. 2017 <https://ui.adsabs.harvard.edu/#abs/2017MNRAS.471.1506K/abstract7>`_ for comparing two data sets with some output of the statistics listed above. It is important to note that few of these distance metrics are defined to be absolute. Rather, most of the metrics give *relative* distances and are defined only when comparing with a common fiducial image.

As shown in `Koch et al. 2017 <https://ui.adsabs.harvard.edu/#abs/2017MNRAS.471.1506K/abstract7>`_, the distance metrics for some statistics have more scatter than others.  Some metrics also suffer from systematic issues and should be avoided when those systematics cannot be controlled for.  The **Cramer distance metric** is an example of this; its shortcomings are described in the paper linked above, and while the implementation is still available, we recommend caution when using it.

The performance of the distance metric for **Tsallis** statistics has not been explored and should also be used with caution.

.. toctree::
    :maxdepth: 2

    metrics/bispectrum_example
    metrics/pca_example
