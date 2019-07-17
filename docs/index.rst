.. TurbuStat documentation master file, created by
   sphinx-quickstart on Mon Jun  8 14:54:35 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TurbuStat
=========

TurbuStat implements a 14 turbulence-based statistics described in the astronomical literature. TurbuStat also defines a distance metrics for each statistic to quantitatively compare spectral-line data cubes, as well as column density, integrated intensity, or other moment maps.

The source code is hosted `here <https://github.com/Astroua/TurbuStat>`_. Contributions to the code base are very much welcome! If you find any issues in the package, please make an `issue on github <https://github.com/Astroua/TurbuStat/issues>`_ or contact the developers at the email on `this page <https://github.com/e-koch>`_. Thank you!

To be notified of future releases and updates to TurbuStat, please join the mailing list: https://groups.google.com/forum/#!forum/turbustat

If you make use of this package in a publication, please cite our accompanying paper::

  @ARTICLE{Koch2019AJ....158....1K,
       author = {{Koch}, Eric W. and {Rosolowsky}, Erik W. and {Boyden}, Ryan D. and
         {Burkhart}, Blakesley and {Ginsburg}, Adam and {Loeppky}, Jason L. and
         {Offner}, Stella S.~R.},
        title = "{TURBUSTAT: Turbulence Statistics in Python}",
      journal = {\aj},
     keywords = {methods: data analysis, methods: statistical, turbulence, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = "2019",
        month = "Jul",
       volume = {158},
       number = {1},
          eid = {1},
        pages = {1},
          doi = {10.3847/1538-3881/ab1cc0},
       eprint = {1904.10484},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019AJ....158....1K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
      }

If your work makes use of the distance metrics, please cite the following::

    @ARTICLE{Koch2017,
     author = {{Koch}, E.~W. and {Ward}, C.~G. and {Offner}, S. and {Loeppky}, J.~L. and {Rosolowsky}, E.~W.},
     title = "{Identifying tools for comparing simulations and observations of spectral-line data cubes}",
     journal = {\mnras},
     archivePrefix = "arXiv",
     eprint = {1707.05415},
     keywords = {methods: statistical, ISM: clouds, radio lines: ISM},
     year = 2017,
     month = oct,
     volume = 471,
     pages = {1506-1530},
     doi = {10.1093/mnras/stx1671},
     adsurl = {http://adsabs.harvard.edu/abs/2017MNRAS.471.1506K},
     adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Citations courtesy of `ADS <https://ui.adsabs.harvard.edu/#>`_


Papers using TurbuStat
----------------------

* `Boyden et al. (2016) <http://adsabs.harvard.edu/abs/2016ApJ...833..233B>`_
* `Koch et al. (2017) <https://ui.adsabs.harvard.edu/#abs/2017arXiv170705415K/abstract>`_
* `Boyden et al. (2018) <https://ui.adsabs.harvard.edu/#abs/2018arXiv180509775B/abstract>`_
* `Sato et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190300764S/abstract>`_
* `Feddersen et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190305104F/abstract>`_


TurbuStat Developers
--------------------

* `Eric Koch <https://github.com/e-koch>`_
* `Erik Rosolowsky <https://github.com/low-sky>`_
* Ryan Boyden
* Blakesley Burkhart
* `Adam Ginsburg <https://github.com/keflavich>`_
* `Jason Loeppky <https://github.com/jloeppky>`_
* Stella Offner
* `Caleb Ward <https://github.com/Astrolebs>`_

Many thanks to everyone who has reported bugs and given feedback on TurbuStat!

* Dario Colombo
* Jesse Feddersen
* Simon Glover
* Jonathan Henshaw
* Sac Medina
* Andrés Izquierdo


Contents:

.. toctree::
   :maxdepth: 2

   install.rst
   accepted_input_formats.rst
   preparing_simulated_data.rst
   data_requirements.rst
   moments.rst
   tutorials/index
   generating_test_data.rst
   statistics.rst
   contributing.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

