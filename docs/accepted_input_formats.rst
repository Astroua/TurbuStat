.. _inputtypes:

*******************
Accepted Data Types
*******************

The TurbuStat routines can accept several different data types.

FITS HDU
*********

The most common format is a FITS HDU. These can be loaded in python with the `~astropy.io.fits` library::

    >>> from astropy.io import fits
    >>> hdulist = fits.open("test.fits")  # doctest: +SKIP
    >>> hdu = hdulist[0]  # doctest: +SKIP

The TurbuStat statistics expect a single extension for a FITS file to be given.

Numpy array and header
**********************

The data can be given as a numpy array, along with a FITS header. This may be useful for data generated from simple physical models (see :ref:`preparing simulated data <simobs_tutorial>`). The data can be given as a 2-element tuple or list::

    >>> input_data = (array, header)  # doctest: +SKIP
    >>> input_data = [array, header]  # doctest: +SKIP

Spectral-Cube objects
*********************

The `spectral-cube <http://spectral-cube.readthedocs.io>`_ package is a dependency of TurbuStat for calculating :ref:`moment arrays <moments>` and spectrally regridding cubes for the :ref:`VCA <vca_tutorial>` statistic. When the data need to be preprocessed, it will often be easiest to work with a SpectralCube object. See the `spectral-cube <http://spectral-cube.readthedocs.io>`_ tutorial for more information::

    >>> from spectral_cube import SpectralCube  # doctest: +SKIP
    >>> cube = SpectralCube("test.fits")  # doctest: +SKIP

This ``SpectralCube`` object can be sliced or used to create moment maps. These spatial 2D maps are called a "Projection" or "Slice" and both are accepted by the TurbuStat statistics::

    >>> sliced_img = cube[100]  # doctest: +SKIP
    >>> moment_img = cube.moment0()  # doctest: +SKIP

The ``Projection`` object also offers a number of convenient functions available for a SpectralCube, making it easy to manipulate and alter the data as needed. To load a spatial FITS image as a projection::

    >>> from spectral_cube import Projection
    >>> img_hdu = fits.open("test_spatial.fits")[0]  # doctest: +SKIP
    >>> proj = Projection.from_hdu(img_hdu)  # doctest: +SKIP

