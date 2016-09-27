
Installing TurbuStat
====================

TurbuStat is currently only available from the `github repo <https://github.com/Astroua/TurbuStat>`_.

TurbuStat requires the follow packages:

 *   astropy>=1.0
 *   numpy>=1.7
 *   matplotlib>=1.2
 *   scipy>=0.12
 *   sklearn>=0.13.0
 *   pandas>=0.13
 *   statsmodels>=0.4.0

The following packages are optional, but required for specific functions in TurbuStat:

 *   `spectral-cube <https://github.com/radio-astro-tools/spectral-cube>`_ - Efficient handling of PPV cubes. Required for calculating moment arrays in `turbustat.data_reduction.Mask_and_Moments`
 *   `astrodendro-development <https://github.com/dendrograms/astrodendro>`_ - Required for calculating dendrograms in `turbustat.statistics.dendrograms`
 *   `emcee <http://dan.iel.fm/emcee/current/>`_ - MCMC fitting in `~turbustat.statistics.PCA`.

When using `turbustat.data_reduction.Mask_and_Moments`, the noise can be automatically estimated by installing two additional packages (**IN DEVELOPMENT**):
 *   `signal-id <https://github.com/radio-astro-tools/signal-id>`_ - Noise estimation in PPV cubes.
 *   `radio_beam <https://github.com/radio-astro-tools/radio_beam>`_ - A class for handling radio beams and useful utilities. Used for noise estimation in signal-id

 To install the packages, clone the repository:
 ::
    >>> git clone https://github.com/Astroua/TurbuStat # doctest: +SKIP

 Then install the package:
 ::
    >>> python setup.py install # doctest: +SKIP

 The script will install numpy and astropy if your python installation does not have them installed. Due to package conflicts, it will **NOT** install the rest of the dependencies! Until this can be fixed, you can check to see if you have all of the dependencies installed by running:
 ::
    >>> python setup.py check_deps # doctest: +SKIP

If you find any issues in the installation, please make an `issue on github <https://github.com/Astroua/TurbuStat/issues>`_ or contact the developers at the email on `this page <https://github.com/e-koch>`_. Thank you!
