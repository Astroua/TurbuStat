TurbuStat
=========

[![Build Status](https://travis-ci.org/Astroua/TurbuStat.svg?branch=master)](https://travis-ci.org/Astroua/TurbuStat)

**NOTE - this package is still under development. API may change!**

*See the documentation at [http://turbustat.readthedocs.org/](http://turbustat.readthedocs.org/)*

Statistics of Turbulence
------------------------

This package is aimed at facilitating comparisons between spectral line data
cubes. Included in this package are several techniques described in the literature
which aim to describe some property of a data cube (or its moments/integrated intensity).
We extend these techniques to be used as comparisons.

Distance Metrics
----------------

Ideally, we require a distance metric to satisfy several properties. A full description
is shown in [Yeremi et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...783...93Y).
The key properties are:
*   cubes with similar physics should have a small distance
*   unaffected by coordinate shifts
*   sensitive to differences in physical scale
*   independent of noise levels in the data

Not all of the metrics satisfy the idealized properties. A full description of all statistics in this package will be shown in Koch et al. (submitted). The paper results can be reproduced using the scripts in [AstroStat_Results](https://github.com/Astroua/AstroStat_Results).

Installing
----------

Currently, the only way install TurbuStat is to clone the repository and run
```python
python setup.py install
```
Due to conflicts with the dependencies, TurbuStat will **NOT** automatically install all
dependencies (only numpy and astropy). To check if your version of python has all the
dependencies installed, run:
```python
python setup.py check_deps
```

Package Dependencies
--------------------

Requires:

 *   astropy>=1.0
 *   numpy>=1.7
 *   matplotlib>=1.2
 *   scipy>=0.12
 *   sklearn>=0.13.0
 *   pandas>=0.13
 *   statsmodels>=0.4.0

Recommended:

 *   [spectral-cube](https://github.com/radio-astro-tools/spectral-cube) - Efficient handling of PPV cubes. Required for calculating moment arrays in `turbustat.data_reduction.Mask_and_Moments`
 *   [astrodendro-development](https://github.com/dendrograms/astrodendro) - Required for calculating dendrograms in `turbustat.statistics.dendrograms`

Optional:
 *   [signal-id](https://github.com/radio-astro-tools/signal-id) - Noise estimation in PPV cubes.
 *   [radio_beam](https://github.com/radio-astro-tools/radio_beam) - A class for handling radio beams and useful utilities. Used for noise estimation in signal-id
 *   [emcee](http://dan.iel.fm/emcee/current/) - Affine Invariant MCMC. Used for fitting the size-line width relation in PCA.


Credits
-------

This package was developed by:

* [Eric Koch](https://github.com/e-koch)
* [Caleb Ward](https://github.com/Astrolebs)
* [Erik Rosolowsky](https://github.com/low-sky)
* [Jason Loeppky](https:/github.com/jloeppky)

If you make use of this package in a publication, please cite our accompanying paper:
```
   @ARTICLE{Koch2016,
    author = {{Koch}, Eric~W. and {Ward}, Caleb~G. and {Offner}, Stella and {Loeppky}, Jason~L. and {Rosolowsky}, Erik~W.},
    title = {Tools for Critically Evaluating Simulations of Star Formation},
    journal = {MNRAS},
    year = {submitted}
    }
```
