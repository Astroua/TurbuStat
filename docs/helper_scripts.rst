
Helper Scripts
==============

The github repository contains several scripts in the `Examples` folder which expand the analysis tools and facilitate running large sets of data. Portions of these scripts will eventually become part of the package itself.

Some scripts are of note for general use with the package:

* `overall_wrapper.py <https://github.com/Astroua/TurbuStat/blob/master/Examples/overall_wrapper.py>`_ allows for quick comparisons of the entire set of statistics. It does not provide the ability to change parameters associated with the statistics, but is useful for initial comparisons. Gaussian noise can be optionally added using the last command line argument. Calling sequence::

    python overall_wrapper.py 1.fits 2.fits T

* `wrapper_2D_data.py <https://github.com/Astroua/TurbuStat/blob/master/Examples/wrapper_2D_data.py>`_ only runs the 2D statistics in the package. Calling sequence::

    python wrapper_2D_data.py 1.fits 2.fits

* The folder `jasper <https://github.com/Astroua/TurbuStat/tree/master/Examples/jasper>`_ contains multiple scripts for use on the Jasper cluster. While they are not intended to be directly portable to other clusters, they provide an example of the basics for running TurbuStat using MPI.

* `analysis_pipeline.py <https://github.com/Astroua/TurbuStat/blob/master/Examples/analysis_pipeline.py>`_ creates analysis plots and performs the modeling for the upcoming paper based on the TurbuStat statistics (Koch et al., 2015 in prep.). The modeling is performed in R and is simply executed from within the script.

* `noise_validation.r <https://github.com/Astroua/TurbuStat/blob/master/Examples/noise_validation.r>`_ and `signal_validation.r <https://github.com/Astroua/TurbuStat/blob/master/Examples/signal_validation.r>`_ test the effectiveness of a statistic given results for a large set of simulations. The noise validation determines whether there is a significant difference between fiducial and design comparisons using a permutation test. Signal validation tests for the amount of scatter between replicated distances (ie. comparing multiple fiducial data to the same design data that has different physical parameters).