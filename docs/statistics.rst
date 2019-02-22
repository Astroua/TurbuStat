
Statistics
==========

Along with acting as the base classes for the distance metrics, the statistics can also return useful information about single datasets.  See the :ref:`runstats` tutorial for more information on computing Statistics.

Distance Metrics
================

The distance metrics are computed using certain outputs contained in the related statistics class.  See the :ref:`runmetrics` tutorial for more information on how to use Distance Metrics.

Nearly all of the distance metrics are actually `"pseudo" - distance metrics <https://en.wikipedia.org/wiki/Pseudometric_space>`_. They must have the following properties:

 1) :math:`d(A, A) = 0`
 2) Symmetric :math:`d(A, B) = d(B, A)`
 3) Triangle Inequality :math:`d(A, B) \leq d(A, C) + d(B, C)`

Here :math:`A` and :math:`B` represent two datasets (either a PPV datacube, column density map, or an associated moment of the cube).

For two datasets with different physical properties, a good statistic will return a large value :math:`d(A, B) \gg 0`. If the datasets have similar physical properties, the distance should be small :math:`d(A, B) \approx 0`. Clear examples of the distance metric properties and the distinction between large and small distances are shown in `Boyden et al. 2016 <https://ui.adsabs.harvard.edu/#abs/2016ApJ...833..233B/abstract>`_ and `Boyden et al. 2018 <https://ui.adsabs.harvard.edu/#abs/2018ApJ...860..157B/abstract>`_.

Additionally, the statistics should ideally be insensitive to spatial shifts :math:`d\left( A\left[ x,y,v \right], A\left[ x+\delta x,y,v \right] \right)=0` and independent of the noise level (for observational data) :math:`d\left( A + \mathcal{N}\left(0, \sigma_1^2 \right), A + \mathcal{N}\left(0, \sigma_2^2 \right) \right) \approx 0`.

Source Code
-----------
.. automodapi:: turbustat.statistics
    :no-inheritance-diagram:
    :inherited-members:
