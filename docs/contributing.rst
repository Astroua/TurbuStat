
Contributing to TurbuStat
=========================

We welcome new contributions to TurbuStat!

To make changes to the source code, make a fork of the `TurbuStat repository <https://github.com/Astroua/TurbuStat>`_ by pressing the button on the top right-hand side of the page. Changes to the code should be made in a new branch, pushed to your fork of the repository, and submitted with a pull request to the main TurbuStat repository. For more information and a detailed guide to using git, please see the `astropy workflow <http://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_ page.

Contributing a new statistic or distance metric
-----------------------------------------------

Increasing the number of statistics implemented in TurbuStat is a key goal for the project. If you plan on providing a new implementation, the contribution should address the following criteria:

1. Statistics and distance metrics should be a `python class <https://docs.python.org/3/tutorial/classes.html>`_. A new folder can be added to the `statistics <https://github.com/Astroua/TurbuStat/tree/master/turbustat/statistics>`_ folder for the new classes. Note that the new folder should contain an `__init__.py` the imports the statistic and/or distance metric classes (see `here <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/pspec_bispec/__init__.py>`__ for an example).

2. Statistics should inherit from the `BaseStatisticMixIn <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/base_statistic.py>`_ class. This mixin class handles unit conversions, extracts the required information from the FITS header, and allows the class to be saved and loaded.

3. Statistic classes should accept the required data products in `__init__`. At a minimum, the class should require the data and a FITS header. Other parameters that define the data or observed object (e.g., distance) can also be specified here. Functions within the class should take the data and compute the statistic.  For example, the `spectral correlation function <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py>`_ (SCF) has separate function to `create the SCF surface <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py#L127>`_, `to make an azimuthally-averaged spectrum <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py#L221>`_, to fit `two-dimensional <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py#L406>`_ and `one-dimensional <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py#L262>`_, and to make a `summary plot <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py#L553>`_.

4. All statistic classes should have a `run` function that computes the statistic and optionally returns a summary plot (e.g., for the `SCF <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py#L725>`_). The `run` function should use sensible defaults, but should have keyword arguments defined for the most important parameters.

5. Distance classes should accept the required data for two data sets in `__init__`. The statistic class should be run on both data sets and keyword arguments should allow the important parameters to be passed to the statistic class (e.g., for the `SCF distance <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py#L806>`_). The distance class should then have a `distance_metric` function defined (`for example <https://github.com/Astroua/TurbuStat/blob/master/turbustat/statistics/scf/scf.py#L862>`_) that computes the distance metric and optionally returns a summary plot of the statistic run on both of the data sets.

6. New statistics and distance metrics should have documentation explaining each argument, keyword argument, parameter, and function in the class. TurbuStat follows the `astropy docstring rules <http://docs.astropy.org/en/stable/development/docrules.html#doc-rules>`_.

7. New statistics and distance metrics must have consistency `tests <https://github.com/Astroua/TurbuStat/tree/master/turbustat/tests>`_. There are two types of consistency tests in TurbuStat: (i) Unit tests against the small test data included in the package (see `here <https://github.com/Astroua/TurbuStat/blob/master/turbustat/tests/_testing_data.py>`__). An output from the statistic and distance metric is saved when the `GetVals.py <https://github.com/Astroua/TurbuStat/blob/master/turbustat/tests/data/GetVals.py>`_ script is run. This type of test determines whether new additions to the code/updates the dependent package change the statistic's output. (ii) Simple model tests where the statistic's output is known. For example, the power-spectrum can be tested by generating an fBM image with a specified index (e.g., see `this test <https://github.com/Astroua/TurbuStat/blob/master/turbustat/tests/test_pspec.py#L106>`_).

8. New statistics and distance metrics should have a tutorial for users (e.g., see `this tutorial <https://github.com/Astroua/TurbuStat/blob/master/docs/tutorials/statistics/scf_example.rst>`_). Ideally, the tutorial will use the data sets already used in the TurbuStat tutorials. If these data are not useful for demonstrating the new statistic, we ask that the tutorial data be (i) small so the computation is relatively quick to run and (ii) publicly-available so users can reproduce the tutorial results.

Questions and feedback
----------------------

Contact the developers at the email on `this page <https://github.com/e-koch>`_ for questions regarding contributions to TurbuStat. We also encourage you to open an pull request even if the code is unfinished (specify "WIP" in the title) and ask the developers for feedback regarding the new contribution.
