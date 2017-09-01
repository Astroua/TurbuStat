
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, erf, erfc
cimport cython

from libc.math cimport sqrt, exp, sin, cos, atan

cdef double pi = np.pi


def C_eps(double r, double k_cut, double alphae, double norm_factor):
    '''

    k_cut is k0 for alphae > 3 and k1 for alphae<3
    '''

    cdef double Int

    # Steep
    if alphae > 3:
        # Eq. B2
        Int = Int3(r, k_cut, alphae)[0]

        return 1 - 4 * pi * Int / norm_factor
    elif alphae == 3:
        # No density dependence.
        return 1. / norm_factor
    # Shallow
    elif alphae > 1:
        # For alphae between 1 and 3

        Int = Int4(r, k_cut, alphae)[0]

        return 1 - 4 * pi * Int / norm_factor

    else:
        raise ValueError("Solution not defined for alphae <= 1.")


def Dz(double r, double theta, double V0, double k0, double alphav):
    '''
    Eq. A3 to A5.

    I've only included the solutions for a solenoidal velocity
    field here. The solution for a potential field is defined in Eqs. A8 & A9
    and can be easily included here. Do they need to be?
    '''

    cdef double intone, inttwo, I_C, I_S

    intone = Int1(r, k0, alphav)[0]
    inttwo = Int2(r, k0, alphav)[0]

    I_C = (4 / 3.) * intone
    I_S = 2 * (inttwo - intone / 3.)

    return 4 * pi * V0**2 * r**(alphav - 3) * \
        (I_C * cos(theta)**2 + I_S * sin(theta)**2)


def F_eps_norm(double alphae, double k_cut):
    '''
    Normalization for F_eps.

    If alphae > 3, k_cut is k0 (Eq. B1). If alphae < 3, k_cut is k1 (Eq. B5).

    '''

    cdef double prefactor

    prefactor = 2 * pi * k_cut**(3 - alphae)

    if alphae > 3:
        return prefactor * gamma(0.5 * (alphae - 3))
    elif alphae < 3:
        return prefactor * gamma(0.5 * (3 - alphae))
    else:
        # If alphae == 3, the density spectrum has no contribution
        return 1.

# Integrals in Dz and C_eps

def Int1(double r, double k0, double alphav):
    '''
    Integral in Eq. A4
    '''
    def integrand(double q):
        cdef double out
        out = q**(2 - alphav) * exp(-(k0 * r / q)**2) * \
            (1 - (3 / q**2) * (sin(q) / q - cos(q)))
        return out

    cdef double value, err

    value, err = quad(integrand, 0, np.inf)

    return value, err


def Int2(double r, double k0, double alphav):
    '''
    First integral in Eq. A5
    '''

    def integrand(double q):
        cdef double out
        out = q**(2 - alphav) * exp(-(k0 * r / q)**2) * \
            (1 - sin(q) / q)
        return out

    cdef double value, err

    value, err = quad(integrand, 0, np.inf)

    return value, err


def Int3(double r, double k0, double alphae):
    '''
    Eq. B2 (w/o 4pi constant)
    '''

    def integrand(double k):
        cdef double out
        out = k**(1 - alphae) * exp(-(k0 / k)**2) * \
            (sin(k * r) / r)
        return out

    cdef double value, err

    value, err = quad(integrand, 0, np.inf)

    return value, err


def Int4(double r, double k1, double alphae):
    '''
    Eq. B7
    '''

    def integrand(double q):
        cdef double out
        out = q**(1 - alphae) * exp(-(q / (k1 * r))**2) * sin(q)
        return out

    cdef double value, err

    value, err = quad(integrand, 0, np.inf)

    return value, err

# Window functions

def gaussian_beam(double theta, double theta_0):
    '''
    Normalized Gaussian

    Note that Eq. 33 and 34 in CL06 are missing the sqrt in the normalization.
    '''
    cdef double out
    out = exp(-(theta / theta_0)**2) / sqrt(pi * theta_0**2)
    return out


def gaussian_autocorr(double R, double z_0, double theta_0):
    '''
    Gaussian autocorrelation function for a circular Gaussian beam defined
    in the projected frame (:math:`\vec{R}` in CL06). This is the solution for
    Eq. 32.

    For W_b:
    :math:`-\frac{e^{-\frac{R^2}{2 \theta ^2 z^2}} \left(\sqrt{\pi } R e^{\frac{R^2}{4 \theta ^2 z^2}} \text{erfc}\left(\frac{R}{2 \theta  z}\right)-2 \theta  z\right)}{4 \theta  z^3}`

    w_b was solved for using Mathematica. My by-hand solution had the same
    asymptotic behaviour, but looked so much worse than the compacter form
    above.

    '''

    cdef double ratio_term, exp_term

    ratio_term = R / (theta_0 * z_0)
    exp_term = ratio_term**2

    term1 = - 1. / (4 * theta_0 * z_0**3)
    term2 = sqrt(pi) * R * exp(- 0.25 * exp_term) * \
        erfc(0.5 * ratio_term)
    term3 = 2 * theta_0 * z_0 * exp(- 0.5 * exp_term)

    return term1 * (term2 - term3)


def slab_autocorr(double z, double z_0, double z_1):
    '''
    Slab model. The function is 1 for z within z_0 and z_1. See Eq. 40.
    This is a normalized version, so the values are 1 / (z_1 - z_0).

    The autocorrelation function is just the square, so
    :math:`(z_1 - z_0)^{-2}`.

    '''

    cdef double out

    if z >= z_0 and z <= z_1:
        out = (z_1 - z_0)**2
        return out


def pencil_beam_gaussian_z(double z, double sigma_z):
    '''
    Pencil beam with a Gaussian w_eps_a.
    '''

    return gaussian_beam(z, 2 * sigma_z)


def pencil_beam_slab_z(double z, double z0, double z1):
    '''
    Pencil beam with a slab w_eps_a.
    '''

    return slab_autocorr(z, z0, z1)


def gaussian_beam_slab_z_crossing(double R, double z, double z0, double z1,
                                  double theta0):
    '''
    Eq. 41 -- Gaussian beam with w_eps as tophat between z0 and z1.

    Same as Eq.14 in Chepurnov+10 (called g(r) there).

    Appropriate for limited resolution with nearby emission.

    '''

    cdef double term1, p, term_a, term_b, term2, term3, term4

    term1 = - 1 / (2 * pi**0.5 * theta0 * R * z)

    p = (z1 + z0) / (z1 - z0)

    term_a = sqrt(2 * z0**2 + p * z**2)
    term_b = sqrt(2 * z1**2 - p * z**2)

    term2 = atan(1 + 2 * z0 / z) + atan(1 - 2 * z1 / z)
    term3 = 1 / term_a - 1 / term_b

    term4 = erf(R / (theta0 * term_a) - erf(R / (theta0 * term_b)))

    return term1 * (term2 / term3) * term4


def gaussian_beam_slab_z_parallel(double R, double z, double z0, double z1,
                                  double theta0):
    '''
    Gaussian beam with w_eps_a as a slab (or tophat) with parallel LOS.

    B/c this is in the parallel limit, the two component for the beam and
    the structure along the LOS are independent. This allows us to easily
    write out the components without needing to solve for the special case,
    as was done for gaussian_beam_slab_z_crossing.

    '''

    return slab_autocorr(z, z0, z1) * gaussian_autocorr(R, z0, theta0)


def gaussian_beam_gaussian_z_parallel(double R, double z, double z_0,
                                      double sigma_z, double theta0):
    '''
    Solution to Eq. 31 assuming a Gaussian beam and a Gaussian for w_eps.

    * Gaussian beam has width theta0 (not FWHM, 2 sigma)
    * Gaussian for structure along LOS at a distance of z0 and a width of
      sigma_z

    The auto-correlation function is presented for both of these.

    For w_eps:
    :math:`e^{-z^2 / 4\sigma_z^2} / \sqrt{4 \pi \sigma_z^2}`

    This is independent of z_0 as it is the distance to the object for this
    definition. The autocorrelation function is for the object thickness along
    the LOS, and so should be independent of distance.

    For W_b:
    :math:`-\frac{e^{-\frac{R^2}{2 \theta ^2 z^2}} \left(\sqrt{\pi } R e^{\frac{R^2}{4 \theta ^2 z^2}} \text{erfc}\left(\frac{R}{2 \theta  z}\right)-2 \theta  z\right)}{4 \theta  z^3}`

    '''

    cdef double w_eps_a, w_b_a

    # See form above. Equal to 2 * sigma_z for "theta_0"
    w_eps_a = gaussian_beam(z, 2 * sigma_z)

    w_b_a = gaussian_autocorr(R, z_0, theta0)

    return w_eps_a * w_b_a
