
Installing TurbuStat
====================

The newest release of TurbuStat is available on pip::

    >>> pip install turbustat  # doctest: +SKIP

TurbuStat can also be installed from the `github repository <https://github.com/Astroua/TurbuStat>`_.

TurbuStat requires the follow packages:

 *   astropy>=2.0
 *   numpy>=1.7
 *   matplotlib>=1.2
 *   scipy>=0.12
 *   sklearn>=0.13.0
 *   statsmodels>=0.4.0
 *   scikit-image>=0.12

The following packages are optional when installing TurbuStat and are required only for specific functions in TurbuStat:

 *   `spectral-cube <https://github.com/radio-astro-tools/spectral-cube>`_ (>v0.4.4) - Efficient handling of PPV cubes. Required for calculating moment arrays in `turbustat.data_reduction.Moments`.
 *   `radio_beam <https://github.com/radio-astro-tools/radio_beam>`_ - A class for handling radio beams and useful utilities. Required for correcting for the beam shape in spatial power spectra. Automatically installed with spectral-cube.
 *   `astrodendro-development <https://github.com/dendrograms/astrodendro>`_ - Required for calculating dendrograms in `turbustat.statistics.dendrograms`
 *   `emcee <http://dan.iel.fm/emcee/current/>`_ - MCMC fitting in `~turbustat.statistics.PCA` and `~turbustat.statistics.PDF`.
 *   `pyfftw <https://hgomersall.github.io/pyFFTW/>`_ - Wrapper for the FFTW libraries. Allows FFTs to be run in parallel.

 To install TurbuStat, clone the repository::
    >>> git clone https://github.com/Astroua/TurbuStat # doctest: +SKIP

Change into the TurbuStat directory and run the following to install TurbuStat::
    >>> python setup.py install # doctest: +SKIP

If you find any issues in the installation, please make an `issue on github <https://github.com/Astroua/TurbuStat/issues>`_ or contact the developers at the email on `this page <https://github.com/e-koch>`_. Thank you!

