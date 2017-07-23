
******************
Tsallis Statistics
******************

Overview
--------

The `Tsallis statistic <https://en.wikipedia.org/wiki/Tsallis_statistics>`_ was introduced by :ref:`Tsallis 1988 <ref-Tsallis1988>` for describing multi-fractal (non-Gaussian) systems. Its use for describing properties of the ISM has been explored in :ref:`Esquivel & Lazarian <ref-Esquivel2010>` and :ref:`Tofflemire et al. 2011 <ref-Tofflemire2011>`. In both of these works, they consider describing an incremental lags in a field by the Tsallis distribution. The specific form of this Tsallis distribution is the `q-Gaussian distribution <https://en.wikipedia.org/wiki/Q-Gaussian_distribution>`_:

.. math::
    R_q = a \left[ 1 + \left( q - 1 \right) \frac{\left[ \Delta f(r) \right]^2}{w^2} \right]^{\frac{-1}{q - 1}}

where :math:`a` is the normalization, :math:`q` controls how "peaked" the distribution is (and is therefore closely related to the kurtosis; :ref:`Moments tutorial <statmoments_tutorial>`), and :math:`w` is the width of the distribution. For :math:`q=1` the distribution reduces to a Gaussian, while :math:`q > 1` gives a flattened distribution with heavier tails. The field is a standardized measure of some quantity: :math:`\Delta f(r) = \left[ f(x, r) - \left< f(x, r) \right>_x \right] / \sqrt{{\rm var}\left[f(x, r)\right]}`, where the angle brackets indicate an average over :math:`x`. The input quantity is the difference over some scale :math:`r` of a field :math:`f(x)`:  :math:`f(x, r) = f(x) - f(x + r)`. The :math:`x, r` are vectors for multi-dimensional data and the formalism is valid for any dimension of data. One distribution is generated for each scale :math:`r`, and the variation of the distribution parameters with changing :math:`r` can be tracked.

Both :ref:`Esquivel & Lazarian <ref-Esquivel2010>` and :ref:`Tofflemire et al. 2011 <ref-Tofflemire2011>` calculate the Tsallis distribution properties for 3D (spatial) and 2D (column density) fields for different sets of simulations. Since TurbuStat is intended to work solely for observable quantities, only the integrated intensity or column density maps can be used. It may useful to investigate other 2D maps using Tsallis statistics, but this has not been explored to date.

Using
-----

**The data in this tutorial are available** `here <https://girder.hub.yt/#user/57b31aee7b6f080001528c6d/folder/59721a30cc387500017dbe37>`_.



References
----------

.. _ref-Tsallis1988:

`Tsallis 1988 <https://link.springer.com/article/10.1007%2FBF01016429>`_

.. _ref-Esquivel2010:

`Esquivel & Lazarian 2010 <https://ui.adsabs.harvard.edu/#abs/2010ApJ...710..125E/abstract>`_

.. _ref-Tofflemire2011:

`Tofflemire et al. 2011 <https://ui.adsabs.harvard.edu/#abs/2011ApJ...736...60T/abstract>`_
