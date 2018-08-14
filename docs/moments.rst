
Deriving Cube Moments
=====================

The `turbustat.moments.Moments` class returns moment arrays as well as their errors for use with 2D statistics in the package. Moments are derived using `spectral-cube <https://github.com/radio-astro-tools/spectral-cube>`_.

Basic Use
---------

Moments are easily returned in the expected form for the statistics::

    >>> from turbustat.moments import Mask_and_Moments  # doctest: +SKIP
    >>> mm = Moments("test.fits")  # doctest: +SKIP
    >>> mm.make_moments()  # doctest: +SKIP
    >>> mm.make_moment_errors()  # doctest: +SKIP
    >>> output_dict = mm.to_dict()  # doctest: +SKIP

`output_dict` now contains the cube and moments along with their respective error maps.
The moments can also be saved::

    >>> mm.to_fits("test")  # doctest: +SKIP

This will return a FITS file for each moment. Error maps are saved in the first extension.

Source Code
-----------
.. automodapi:: turbustat.moments
    :no-heading:
    :no-inheritance-diagram:
