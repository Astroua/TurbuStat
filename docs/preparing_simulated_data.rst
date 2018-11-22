.. _simobs_tutorial:

************************
Preparing Simulated Data
************************

TurbuStat requires the input data be in a valid FITS format. Since simulated observations do not always include a valid observational FITS header, we provide a few convenience functions to create a valid format.

We start with a numpy array of data from some source. First consider a PPV cube. We will need to specify several quantities, like the angular pixel scale, to create the header:

    >>> import numpy as np
    >>> import astropy.units as u
    >>> from turbustat.io.sim_tools import create_fits_hdu
    >>> cube = np.ones((8, 16, 16))
    >>> pixel_scale = 0.001 * u.deg
    >>> spec_pixel_scale = 1000. * u.m / u.s
    >>> beamfwhm = 0.003 * u.deg
    >>> imshape = cube.shape
    >>> restfreq = 1.4 * u.GHz
    >>> bunit = u.K
    >>> cube_hdu = create_fits_hdu(cube, pixel_scale, spec_pixel_scale, beamfwhm, imshape, restfreq, bunit)


`cube_hdu` can now be passed to the TurbuStat statistics, or loaded into a `spectral_cube.SpectralCube` with `SpectralCube.read(cube_hdu)` for easy manipulation of the PPV cube.

For a two-dimensional image, the FITS HDU can be made in almost the same way, minus `spec_pixel_scale`:

    >>> img = np.ones((16, 16))
    >>> imshape = img.shape
    >>> img_hdu = create_fits_hdu(img, pixel_scale, beamfwhm, imshape, restfreq, bunit)

The FITS HDU can be given to TurbuStat statistics, or converted to a `spectral_cube.Projection` with `Projection.from_hdu(img_hdu)`.

You can also create just the FITS headers with:

    >>> from turbustat.io.sim_tools import create_image_header, create_cube_header
    >>> img_hdr = create_image_header(pixel_scale, beamfwhm, img.shape, restfreq, bunit)
    >>> cube_hdr = create_cube_header(pixel_scale, spec_pixel_scale, beamfwhm, cube.shape, restfreq, bunit)


Units
*****

The units should be an equivalent observational unit, depending on the required data product (PPV cube, zeroth moment, centroid, etc...). While TurbuStat does not explicitly check for the input data units, two things should be kept in mind:

1. The data cannot be log-scaled.

2. When comparing data sets, both should have the same unit. Most statistics are not based on the absolute scale in the data, but it is best to avoid possible misinterpretation of the results.


Source Code
-----------
.. automodapi:: turbustat.io.sim_tools
    :no-inheritance-diagram:
    :inherited-members:
