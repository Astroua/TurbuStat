
All statistics and distance metrics are contained in the ```turbustat.statistics``` module.

Statistics
==========

Along with acting as the base classes for the distance metrics, the statistics can also return useful information about single datasets.

Distance Metrics
================

The distance metrics are computed using certain outputs contained in the related statistics class.

Nearly all of the distance metrics are actually ["pseudo" - distance metrics](https://en.wikipedia.org/wiki/Pseudometric_space). They must have the following properties:

 1) :math:`d(A, A) = 0`
 2) Symmetric :math:`d(A, B) = d(B, A)`
 3) Triangle Inequality :math:`d(A, B) \leq d(A, C) + d(B, C)`

.. toctree::
   :maxdepth: 2

   statistics/bispectrum.rst
   statistics/cramer.rst
   statistics/deltavariance.rst
   statistics/dendrograms.rst
   statistics/genus.rst
   statistics/highstats.rst
   statistics/mahalanobis.rst
   statistics/mvc.rst
   statistics/pca.rst
   statistics/pdf.rst
   statistics/pspec.rst
   statistics/scf.rst
   statistics/tsallis.rst
   statistics/vca.rst
   statistics/vcs.rst
   statistics/wavelets.rst
