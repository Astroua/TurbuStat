
Deriving Cube Moments
=====================

The `Mask_and_Moments`_ class returns moment arrays as well as their errors for use with 2D statistics in the package. Noise is estimated using `signal_id <https://github.com/radio-astro-tools/signal-id>`_. Moments are derived using `spectral-cube <https://github.com/radio-astro-tools/spectral-cube>`_ and are able to handle massive datacubes.

Basic Use
---------

Moments are easily returned in the expected form for the statistics:
::
    from turbustat.data_reduction import Mask_and_Moments

    mm = Mask_and_Moments("test.fits")
    mm.make_moments()
    mm.make_moment_errors()
    output_dict = mm.to_dict()

`output_dict` now contains the cube and moments along with their respective error maps.
The moments can also be saved:
::
    mm.to_fits("test")

This will return a FITS file for each moment. Error maps are saved in the first extension.

Source Code
-----------
.. automodapi:: turbustat.data_reduction
    :no-heading:
    :no-inheritance-diagram:
