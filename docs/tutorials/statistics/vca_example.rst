
.. _vca_tutorial:

*******************************
Velocity Channel Analysis (VCA)
*******************************

Overview
--------

A major advantage of a spectral-line data cube, rather than an integrated two-dimensional image, is that it captures aspects of both the density and velocity fluctuations in the field of observation. :ref:`Lazarian & Pogosyan 2000 <_ref-lp00>` and :ref:`Lazarian & Pogosyan 2004 <_ref-lp04>` derived how the power spectrum from a cube depends on the statistics of the density and velocity fields for the 21-cm Hydrogen line, allowing for each their properties to be examined (provided the data has sufficient spectral resolution).

The Lazarian & Pogosyan theory predicts two regimes based on the the power-spectrum slope: the *shallow* (:math:`n < -3`) and the *steep* (:math:`n < -3`) regimes. In the case of optically thick line emission, :ref:`Lazarian & Pogosyan 2004 <_ref-lp04>` show that the slope saturates to :math:`n = -3` (see :ref:`Burkhart et al. 2013 <_ref-burkhart2013>` as well). The VCA predictions in these different regimes are shown in Table 1 of :ref:`Chepurnov & Lazarian 2009 <_ref-chepurnov09>` (also see Table 3 in :ref:`Lazarian 2009 <_ref-lazarian09>`). The complementary :ref:`Velocity Coordinate Spectrum <vca_tutorial>` can be used in tandem with VCA.

Using
-----

**The data in this tutorial are available** `here <https://girder.hub.yt/#user/57b31aee7b6f080001528c6d/folder/59721a30cc387500017dbe37>`_.

We need to import the `~turbustat.statistics.VCA` class, along with a few other common packages:

    >>> from turbustat.statistics import VCA
    >>> from astropy.io import fits

And we load in the data cube:

    >>> cube = fits.open("Design4_flatrho_0021_00_radmc.fits")[0]  # doctest: +SKIP

The VCA spectrum is computed using:

    >>> vca = VCA(cube)  # doctest: +SKIP
    >>> vca.run(verbose=True)  # doctest: +SKIP
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.973
    Model:                            OLS   Adj. R-squared:                  0.973
    Method:                 Least Squares   F-statistic:                     3188.
    Date:                Thu, 20 Jul 2017   Prob (F-statistic):           1.75e-71
    Time:                        15:14:32   Log-Likelihood:                -1.2719
    No. Observations:                  91   AIC:                             6.544
    Df Residuals:                      89   BIC:                             11.57
    Df Model:                           1
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.3928      0.058     41.036      0.000       2.277       2.509
    x1            -4.2546      0.075    -56.459      0.000      -4.404      -4.105
    ==============================================================================
    Omnibus:                        4.747   Durbin-Watson:                   0.069
    Prob(Omnibus):                  0.093   Jarque-Bera (JB):                4.622
    Skew:                          -0.550   Prob(JB):                       0.0992
    Kurtosis:                       2.916   Cond. No.                         4.40
    ==============================================================================

.. image:: images/design4_vca.png

The VCA power spectrum from this simulated data cube is :math:`-4.25\pm0.08`, which is steeper than the power spectrum we found using the zeroth moment (:ref:`PowerSpectrum tutorial <pspec_tutorial>`). However, as was the case for the power-spectrum of the zeroth moment, there are deviation from a single power-law on small scales due to the inertial range in the simulation. The spatial frequencies used in the fit can be limited by setting `low_cut` and `high_cut`. The inputs should have frequency units in pixels, angle, or physical units. In this case, we will limit the fitting between frequencies of `0.02 / pix` and `0.1 / pix` (where the conversion so pixel scales in the simulation is just `1 / freq`):

    >>> vca.run(verbose=True, xunit=u.pix**-1, low_cut=0.02 / u.pix, high_cut=0.1 / u.pix)  # doctest: +SKIP
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.985
    Model:                            OLS   Adj. R-squared:                  0.984
    Method:                 Least Squares   F-statistic:                     866.6
    Date:                Thu, 20 Jul 2017   Prob (F-statistic):           2.77e-13
    Time:                        15:28:29   Log-Likelihood:                 17.850
    No. Observations:                  15   AIC:                            -31.70
    Df Residuals:                      13   BIC:                            -30.28
    Df Model:                           1
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          3.7695      0.134     28.031      0.000       3.479       4.060
    x1            -3.0768      0.105    -29.438      0.000      -3.303      -2.851
    ==============================================================================
    Omnibus:                        1.873   Durbin-Watson:                   2.409
    Prob(Omnibus):                  0.392   Jarque-Bera (JB):                1.252
    Skew:                          -0.684   Prob(JB):                        0.535
    Kurtosis:                       2.641   Cond. No.                         13.5
    ==============================================================================

.. image:: images/design4_vca_limitedfreq.png

With the fitting limited to the valid region, we find a shallower slope of :math:`-3.1\pm0.1` and a better fit to the model. `low_cut` and `high_cut` can also be given as spatial frequencies in angular units (e.g., `u.deg**-1`). When a distance is given, the `low_cut` and `high_cut` can also be given in physical frequency units (e.g., `u.pc**-1`).

Breaks in the power-law behaviour in observations (and higher-resolution simulations) can result from differences in the physical processes dominating at those scales. To capture this behaviour, `VCA` can be passed a break point to enable fitting with a segmented linear model (`~turbustat.statistics.Lm_Seg`; see the description given in the :ref:`PowerSpectrum tutorial <pspec_tutorial>`). In this example, we will assume a distance of 250 pc in order to show the power spectrum in physical units:

    >>> vca = VCA(cube, distance=250 * u.pc)  # doctest: +SKIP
    >>> vca.run(verbose=True, xunit=u.pc**-1, low_cut=0.02 / u.pix, high_cut=0.4 / u.pix, brk=0.1 / u.pix, log_break=False)  # doctest: +SKIP
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.998
    Model:                            OLS   Adj. R-squared:                  0.998
    Method:                 Least Squares   F-statistic:                 1.113e+04
    Date:                Thu, 20 Jul 2017   Prob (F-statistic):           2.66e-90
    Time:                        16:19:33   Log-Likelihood:                 101.91
    No. Observations:                  71   AIC:                            -195.8
    Df Residuals:                      67   BIC:                            -186.8
    Df Model:                           3
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          3.6333      0.053     68.784      0.000       3.528       3.739
    x1            -3.1814      0.047    -67.916      0.000      -3.275      -3.088
    x2            -2.4558      0.094    -26.152      0.000      -2.643      -2.268
    x3            -0.0097      0.027     -0.355      0.724      -0.065       0.045
    ==============================================================================
    Omnibus:                        8.205   Durbin-Watson:                   1.148
    Prob(Omnibus):                  0.017   Jarque-Bera (JB):                7.707
    Skew:                          -0.772   Prob(JB):                       0.0212
    Kurtosis:                       3.469   Cond. No.                         20.8
    ==============================================================================

.. image:: images/design4_vca_breakfit.png

By incorporating the break, we find a better quality fit to this portion of the power-spectrum. We also find that the, for the slope before the break (i.e., in the inertial range), the slope is consistent with the slope from the zeroth moment (:ref:`PowerSpectrum tutorial <pspec_tutorial>`). The break point was moved significantly from the initial guess, which we had set to the upper limit of the inertial range:

    >>> vca.brk  # doctest: +SKIP
    <Quantity 0.1624771454997838 1 / pix>
    >>> vca.brk_err  # doctest: +SKIP
    <Quantity 0.010241094948585336 1 / pix>

From the figure, this is where the curve deviates from the power-law on small scales. With our assigned distance, the break point corresponds to a physical scale of:

    >>> vca._physical_size / vca.brk.value
    <Quantity 0.14082499334584425 pc>

`vca._physical_size` is the spatial size of one pixel (assuming the spatial dimensions have square pixels in the celestial frame).

The values of the slope after the break point (`x2`) in the fit description is defined relative to the first slope. Its actual slope would then be the sum of `x1` and `x2`. The slopes and their uncertainties can be accessed through:

    >>> vca.slope  # doctest: +SKIP
    array([-3.18143757, -5.63724147])
    >>> vca.slope_err  # doctest: +SKIP
    array([ 0.04684344,  0.104939  ])

The slope above the break point is within the uncertainty of the slope we found in the second example (:math:`-3.1\pm0.1`). The uncertainty we find here is nearly half of the previous one since more points have been used in this fit.

The Lazarian & Pogosyan theory predicts that the VCA power-spectrum depends on the size of the velocity slices in the data cube (e.g., :ref:`Stanimirovic & Lazarian 2001 <ref-sl01>`). `~turbustat.statistics.VCA` allows for the velocity channel thickness to be changed with `channel_width`. This runs a routine that spectrally smooths the cube with a Gaussian kernel, whose width matched the target `channel_width`, then interpolates the data onto a new grid at the new `channel_width`. The example data used here has spectral channels of :math:`\sim 40` m / s. We can re-run VCA on this data with a channel width of :math:`\sim 400` m / s, and compare to the original slope:

    >>> vca_thicker_channel = VCA(cube, distance=250 * u.pc, channel_width=400 * u.m / u.s)  # doctest: +SKIP
    >>> vca_thicker.run(verbose=True, xunit=u.pc**-1, low_cut=0.02 / u.pix, high_cut=0.4 / u.pix, brk=0.1 / u.pix, log_break=False)  # doctest: +SKIP
                           OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.998
    Model:                            OLS   Adj. R-squared:                  0.998
    Method:                 Least Squares   F-statistic:                     9739.
    Date:                Thu, 20 Jul 2017   Prob (F-statistic):           2.29e-88
    Time:                        19:00:25   Log-Likelihood:                 94.310
    No. Observations:                  71   AIC:                            -180.6
    Df Residuals:                      67   BIC:                            -171.6
    Df Model:                           3
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          1.4422      0.057     25.516      0.000       1.329       1.555
    x1            -3.2388      0.051    -64.014      0.000      -3.340      -3.138
    x2            -2.8668      0.108    -26.651      0.000      -3.081      -2.652
    x3             0.0116      0.030      0.385      0.702      -0.049       0.072
    ==============================================================================
    Omnibus:                        7.262   Durbin-Watson:                   1.043
    Prob(Omnibus):                  0.026   Jarque-Bera (JB):                6.646
    Skew:                          -0.720   Prob(JB):                       0.0361
    Kurtosis:                       3.418   Cond. No.                         20.9
    ==============================================================================

.. image:: images/design4_vca_400ms_channels.png

With the original spectral resolution, the slope in the inertial range was already consistent with the "thickest slice" case, the zeroth moment. The slope here remains consistent with the zeroth moment power-spectrum, so for this data set of :math:`^{13}{\rm CO}`, there is no evolution in the spectrum with channel size.


References
----------

.. _ref-lp00:

`Lazarian & Pogosyan 2000 <https://ui.adsabs.harvard.edu/#abs/2000ApJ...537..720L/abstract>`_

.. _ref-lp04:

`Lazarian & Pogosyan 2004 <https://ui.adsabs.harvard.edu/#abs/2004ApJ...616..943L/abstract>`_

.. _ref-sl01:

`Stanimirovic & Lazarian 2001 <https://ui.adsabs.harvard.edu/#abs/2001ApJ...551L..53S/abstract>`_

.. _ref-burkhart2013:

`Burkhart et al. 2013 <https://ui.adsabs.harvard.edu/#abs/2013ApJ...771..123B/abstract>`_

.. _ref-chepurnov09:

`Chepurnov & Lazarian 2009 <https://ui.adsabs.harvard.edu/#abs/2009ApJ...693.1074C/abstract>`_

.. _ref-lazarian09:

`Lazarian 2009 <https://ui.adsabs.harvard.edu/#abs/2009SSRv..143..357L/abstract>`_
