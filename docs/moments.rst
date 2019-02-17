.. _moments:

Deriving Cube Moments
=====================

The `~turbustat.moments.Moments` class returns moment arrays as well as their errors for use with 2D statistics in the package. Moments are derived using `spectral-cube <https://github.com/radio-astro-tools/spectral-cube>`_. Definitions for the moments are also available in the `spectral-cube documentation <https://spectral-cube.readthedocs.io/en/latest/api/spectral_cube.SpectralCube.html#spectral_cube.SpectralCube.moment>`_.

Basic Use
---------

Moments are easily returned in the expected form for the statistics with `~turbustat.moments.Moments`. This class takes a FITS file of a spectral-line cube as input and creates moment arrays (zeroth, first, and line width) with their respective uncertainties.::

    >>> from turbustat.moments import Moments  # doctest: +SKIP
    >>> # Load in the cube "test.fits"
    >>> mm = Moments("test.fits")  # doctest: +SKIP
    >>> mm.make_moments()  # doctest: +SKIP
    >>> mm.make_moment_errors()  # doctest: +SKIP
    >>> output_dict = mm.to_dict()  # doctest: +SKIP

`output_dict` is a dictionary that contains keys for the cube and moments. The moment keys contain a list of the moment and its error map.

The moments can also be saved to FITS files. The `~turbustat.moments.Moments.to_fits` function saves FITS files of each moment. The input to the function is the prefix to use in the filenames::

    >>> mm.to_fits("test")  # doctest: +SKIP

This will produce three FITS files: `test_moment0.fits`, `test_centroid.fits`, `test_linewidth.fits` for the zeroth, first, and square-root of the second moments, respectively. These FITS files will contain two extensions, the first with the moment map and the second with the uncertainty map for that moment.

Source Code
-----------
.. automodapi:: turbustat.moments
    :no-heading:
    :no-inheritance-diagram:
