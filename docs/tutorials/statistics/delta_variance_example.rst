
**************
Delta-Variance
**************

Overview
--------

The :math:`\Delta`-variance technique was introduced by :ref:`Stutzki et al. 1998 <_ref-stutzki1998>` as a generalization of *Allan-variance*, a technique used to study the drift in instrumentation. They found that molecular cloud structures are well characterized by fractional Brownian motion structure, which results from a power-law power spectrum with a random phase distribution. The technique was extended by :ref:`Bensch et al. 2001 <_ref-bensch2001>` to account for the effects of beam smoothing and edge effects on a discretely sampled grid. With this, they identified a functional form to recover the index of the near power-law relation. The technique was extended again by :ref:`Ossenkopf et al. 2008a <_ref-ossenkopf2008a>`, where the computation using filters of different scales was moved in to the Fourier domain, allowing for a significant improvement in speed. The following description uses their formulation.

Delta-variance measures the amount of structure on a given range of scales. Each delta-variance point is calculated by filtering an image with a spherically symmetric kernel - a French-hat or Mexican hat (Ricker) kernels - and computing the variance of the filtered map. Due to the effects of a finite grid that typically does not have periodic boundaries and the effects of noise, :ref:`Ossenkopf et al. 2008a <_ref-ossenkopf2008a>` proposed a convolution based method that splits the kernel into its central peak and outer annulus, convolves the separate regions, and subtracts the annulus-convolved map from the peak-convolved map. The Mexican-hat kernel separation can be defined using two Gaussian functions. A weight map was also introduced to minimize noise effects where there is low S/N regions in the data. Altogether, this is expressed as:

.. math::
    F(r) = \frac{G_{\rm core}(r)}{W_{\rm core}(r)} - \frac{G_{\rm ann}(r)}{W_{\rm ann}(r)}

where :math:`r` is the kernel size, :math:`G` is the convolved image map, and :math:`W` is the convolved weight map. The delta-variance is then,

.. math::
    \sigma_{\delta}(r) = \frac{\Sigma_{\rm map} \mathrm{Var}(F(r)) W_{\rm tot}(r)}{\Sigma_{\rm map} W_{\rm tot}(r)}

where :math:`W_{\rm tot}(r) = W_{\rm core}(r)\,W_{\rm ann}(r)`.

Since the kernel is separated into two components, the ratio between their widths can be set independently. :ref:`Ossenkopf et al. 2008a <_ref-ossenkopf2008a>` find an optimal ratio of 1.5 for the Mexican-hat kernel, which is the element used in TurbuStat.

Performing this operation yields a power-law-like relation between the scales :math:`r` and the delta-variance.

Using
-----

**The data in this tutorial are available** `here <https://girder.hub.yt/#user/57b31aee7b6f080001528c6d/folder/57e55670a909a80001d301ae>`_.

We need to import the `~turbustat.statistics.DeltaVariance` code, along with a few other common packages:

    >>> from turbustat.statistics import DeltaVariance
    >>> from astropy.io import fits

Then, we load in the data and the associated error array:

    >>> moment0 = fits.open("Design4_21_0_0_flatrho_0021_13co.moment0.fits")[0]  # doctest: +SKIP
    >>> moment0_err = fits.open("Design4_21_0_0_flatrho_0021_13co.moment0_error.fits")[0]  # doctest: +SKIP

Next, we initialize the `~turbustat.statistics.DeltaVariance` class:

    >>> delvar = DeltaVariance(moment0, weights=moment0_err)  # doctest: +SKIP

The weight array is optional, but is recommended. Note that this is not the exact form of a weight array used by :ref:`Ossenkopf et al. 2008b <_ref-ossenkopf2008b>`, who used the sqrt of the number of elements along the line of sight used to create the integrated intensity map. This doesn't take into account the varying S/N of each element used, however. In the case with the simulated data, the two are nearly identical, since the noise value associated with each element is constant. If no weights are given, a uniform array of ones is used.

By default, 25 lag values will be used, logarithmically spaced between 3 pixels to half of the minimum axis size. Alternative lags can be specified by setting the `lags` keyword. If a `~numpy.ndarray` is passed, it is assumed to be in pixel units. Lags can also be given in angular units.

The entire process is performed through `~turbustat.statistics.DeltaVariance.run`:

    >>> delvar.run(verbose=True, ang_units=True, unit=u.arcmin)  # doctest: +SKIP

.. image:: images/delvar_design4.png

`ang_units` sets showing angular scales in the plot, and `unit` is the unit to show them in.

References
----------

.. _ref-stutzki1998:

`Stutzki et al. 1998 <https://ui.adsabs.harvard.edu/#abs/1998A&A...336..697S/abstract>`_

.. _ref-bensch2001:

`Bensch, F. <https://ui.adsabs.harvard.edu/#abs/2001A&A...366..636B/abstract>`_

.. _ref-ossenkopf2008a:

`Ossenkopf at al. 2008a <https://ui.adsabs.harvard.edu/#abs/2008A&A...485..917O/abstract>`_

.. _ref-ossenkopf2008b:

`Ossenkopf at al. 2008b <https://ui.adsabs.harvard.edu/#abs/2008A&A...485..719O/abstract>`_
