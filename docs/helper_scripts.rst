
Helper Scripts
==============

The github repository contains several scripts in the `Examples` folder which expand the analysis tools and facilitate running large sets of data. Portions of these scripts will eventually become part of the package itself.

Some scripts are of note for general use with the package:

* `overall_wrapper.py <https://github.com/Astroua/TurbuStat/blob/master/Examples/overall_wrapper.py>`_ allows for quick comparisons of the entire set of statistics. It does not provide the ability to change parameters associated with the statistics, but is useful for initial comparisons. Gaussian noise can be optionally added using the last command line argument. Calling sequence::

    python overall_wrapper.py 1.fits 2.fits T

* `wrapper_2D_data.py <https://github.com/Astroua/TurbuStat/blob/master/Examples/wrapper_2D_data.py>`_ only runs the 2D statistics in the package. Calling sequence::

    python wrapper_2D_data.py 1.fits 2.fits
