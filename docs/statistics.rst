
Statistics
==========

Along with acting as the base classes for the distance metrics, the statistics can also return useful information about single datasets.

Distance Metrics
================

The distance metrics are computed using certain outputs contained in the related statistics class.

Nearly all of the distance metrics are actually `"pseudo" - distance metrics <https://en.wikipedia.org/wiki/Pseudometric_space>`_. They must have the following properties:

 1) :math:`d(A, A) = 0`
 2) Symmetric :math:`d(A, B) = d(B, A)`
 3) Triangle Inequality :math:`d(A, B) \leq d(A, C) + d(B, C)`


Source Code
-----------
.. automodapi:: turbustat.statistics
    :no-heading:
    :no-inheritance-diagram:
