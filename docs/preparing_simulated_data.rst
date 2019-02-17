.. _simobs_tutorial:

************************
Preparing Simulated Data
************************

TurbuStat requires the input data be in a valid FITS format. Since simulated observations do not always include a valid observational FITS header, we provide a few convenience functions to create a valid format.

We start with a numpy array of data from some source. First consider a spectral-line data cube with 2 spatial dimensions and one spectral dimension (also called a PPV cube; position-position-velocity). We will need to specify several quantities, like the angular pixel scale, to create the header:

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


Noise and missing data
**********************

To create realistic data, noise could be added to `img` and `cube` in the above examples.  The simplest form of noise is Gaussian and can be added to the data with::

    >>> sigma = 0.1
    >>> noisy_cube = cube + np.random.normal(0., sigma, size=cube.shape)

In this example, Gaussian noise with standard deviation of 0.1 and mean of 0. is added to the cube.

A common observational practice is to mask noise-only regions of an image or data cube.  A simple example is, with knowledge of the standard deviation of the noise, to impose an :math:`N-\sigma` cut to the data::

    >>> N = 3.
    >>> # Create a boolean array, where True is above the noise threshold.
    >>> signal_mask = noisy_cube > N * sigma
    >>> # Set all places below the noise threshold to NaN in the cube
    >>> masked_cube = signal_mask.copy()
    >>> masked_cube[~signal_mask] = np.NaN

TurbuStat does not contain routines to create robust signal masks. Examples of creating signal masks can be found in `Rosolowsky & Leroy 2006 <https://ui.adsabs.harvard.edu/#abs/2006PASP..118..590R/abstract>`_ and `Dame 2011 <https://ui.adsabs.harvard.edu/#abs/2011arXiv1101.1499D/abstract>`_.

References
----------

`Dame 2011 <https://ui.adsabs.harvard.edu/#abs/2011arXiv1101.1499D/abstract>`_

`Rosolowsky & Leroy 2006 <https://ui.adsabs.harvard.edu/#abs/2006PASP..118..590R/abstract>`_

Excluding the dissipation range
*******************************

When using synthetic observations from simulations, care should be taken to only fit scales in the inertial range. The :ref:`power-spectrum tutorial <pspec_tutorial>` shows an example of limiting the fit to the inertial range.  The power-spectrum in the dissipation range in that example steepens significantly and is not representative of the turbulent index.  This warning should be heeded for power-spectrum-based methods, like the spatial power-spectrum, MVC, VCA and VCS. Spatial structure functions, like the wavelet transform and the delta-variance should also be examined closely for the inflence of the dissipation range on small scales.


Source Code
-----------
.. automodapi:: turbustat.io.sim_tools
    :no-inheritance-diagram:
    :inherited-members:
