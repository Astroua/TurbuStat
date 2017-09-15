
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, erf, erfc, hyp1f1


import flint
hyper = flint.arb.hypgeom

# ImportError = Exception

# try:
#     import flint
#     hyper = flint.arb.hypergeom
# except ImportError:
#     from mpmath import hyper

cimport cython

from libc.math cimport sqrt, exp, sin, cos, atan

cdef double pi = np.pi


def C_eps(double r, double k_cut, double alphae, double norm_factor):
    '''

    k_cut is k0 for alphae > 3 and k1 for alphae<3


    When alphae<3, this uses the analytic solution to Eq. B7,

    :math:`\int_0^\inf q^{1 - \alpha_e} \sin(q) \exp[-q^2/(k_1 r)^2]`

    :math:`0.5 (k_1 r)^{3 - \alpha_e} \Gamma(0.5 * (3 - alpha_e) {\rm F1}(0.5 (3 - alphae), 3/2, -0.25 (k_1 r)^2))`

    '''

    cdef double Int, kr, nu

    # Steep
    if alphae > 3:
        # Eq. B2
        # Int = Int3(r, k_cut, alphae)[0]
        Int = Int3_analytic(r, k_cut, alphae)

        return 1 + 4 * pi * r**(alphae - 3) * Int / norm_factor
    elif alphae == 3:
        # No density dependence.
        return 1.
    # Shallow
    elif alphae > 1:
        # For alphae between 1 and 3. See full solution above and in B7.
        # The form is simplified here for efficiency

        nu = 3 - alphae
        kr = k_cut * r

        return 1 + 2 * pi * k_cut**nu * gamma(0.5 * nu) * \
            hyp1f1(0.5 * nu, 1.5, -0.25 * kr**2) / norm_factor
    else:
        raise ValueError("Solution not defined for alphae <= 1.")


def Dz(double R, double z, double V0, double k0, double alphav):
    '''
    Eq. A3 to A5.

    I've only included the solutions for a solenoidal velocity
    field here. The solution for a potential field is defined in Eqs. A8 & A9
    and can be easily included here. Do they need to be?
    '''

    cdef double intone, inttwo, I_C, I_S, costheta, sintheta, r

    r = np.sqrt(R**2 + z**2)

    # intone = Int1(r, k0, alphav)[0]
    # inttwo = Int2(r, k0, alphav)[0]

    intone = Int1_analytic(r, k0, alphav)
    inttwo = Int2_analytic(r, k0, alphav)

    I_C = (4 / 3.) * intone
    I_S = 2 * (inttwo - intone / 3.)

    costheta = z / r
    sintheta = R / r

    return 4 * pi * V0**2 * r**(alphav - 3) * \
        (I_C * costheta**2 + I_S * sintheta**2)


def Dz_simp(double R, double z, double V0, double k0, double alphav):

    cdef double costheta, sintheta, r

    r = np.sqrt(R**2 + z**2)

    return 4 * pi * V0**2 * r**(alphav - 3) / (r**(alphav - 3) + k0**(3 - alphav))


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
            (1 - (3 / q**2) * ((sin(q) / q) - cos(q)))
        return out

    cdef double value, err

    value, err = quad(integrand, 0, np.inf)

    return value, err


def Int1_analytic(double r, double k0, double alphav):
    '''
    Analytical solution to Eq. A4
    '''

    cdef double kr, kr2, nu, term1, term2, term3, term4, terma, termb

    kr2 = (k0 * r)**2
    kr = k0 * r
    nu = alphav - 3

    term1 = 3 * nu * float(hyper([], [0.5, 1.5 - 0.5 * alphav], 0.25 * kr2))
    term2 = 3 * nu * float(hyper([], [1.5, 1.5 - 0.5 * alphav], 0.25 * kr2))
    terma = 0.25 * kr**(1 - alphav) * gamma(0.5 * nu) * (2 * kr2 + term1 - term2)

    term3 = float(hyper([], [0.5 * (1 + alphav), 1 + 0.5 * alphav], 0.25 * kr2))
    term4 = alphav * float(hyper([], [0.5 * (1 + alphav), 0.5 * alphav], 0.25 * kr2))
    termb = 3 * gamma(-alphav) * sin(alphav * pi * 0.5) * (term3 - term4)

    return terma + termb


def Int2(double r, double k0, double alphav):
    '''
    First integral in Eq. A5
    '''

    def integrand(double q):
        cdef double out
        out = q**(2 - alphav) * exp(-(k0 * r / q)**2) * \
            (1 - (sin(q) / q))
        return out

    cdef double value, err

    value, err = quad(integrand, 0, np.inf)

    return value, err


def Int2_analytic(double r, double k0, double alphav):
    '''
    Analytical solution to first integral in Eq. A5
    '''

    cdef double kr, kr3, kr2, term1, term2, term3, nu

    kr3 = (k0 * r)**3
    kr2 = (k0 * r)**2
    kr = k0 * r
    nu = alphav - 3


    term1 = - kr3 * gamma(0.5 * nu)
    term2 = kr3 * gamma(0.5 * nu) * float(hyper([], [1.5, 2.5 - 0.5 * alphav], 0.25 * kr2))
    term3 = 2 * kr**alphav * gamma(2 - alphav) * float(hyper([], [0.5 * (alphav - 1), 0.5 * alphav], 0.25 * kr2)) * sin(alphav * pi * 0.5)

    return -0.5 * kr**-alphav * (term1 + term2 + term3)


def Int3(double r, double k0, double alphae):
    '''
    Eq. B2 (w/o 4pi constant)
    '''

    def integrand(double q):
        cdef double out
        out = q**(1 - alphae) * exp(-(k0 * r / q)**2) * sin(q)
        return out

    cdef double value, err

    value, err = quad(integrand, 0, np.inf)

    return value, err


def Int3_analytic(double r, double k0, double alphae):
    '''

    Analytic solution for Eq. B2

    '''

    cdef double term1, term2, nu, kr2

    nu = 3 - alphae
    kr2 = (k0 * r)**2

    term1 = 0.5 * (k0 * r)**nu * gamma(- 0.5 * nu) * float(hyper([], [1.5, 2.5 - 0.5 * alphae], 0.25 * kr2))
    term2 = gamma(2 - alphae) * float(hyper([], [0.5 * (- 1 + alphae), 0.5 * alphae], 0.25 * kr2)) * sin(0.5 * alphae * pi)

    return term1 + term2

def Int4(double r, double k1, double alphae):
    '''
    Analytic solution to Eq. B7,

    :math:`\int_0^\inf q^{1 - \alpha_e} \sin(q) \exp[-q^2/(k_1 r)^2]`

    :math:`0.5 (k_1 r)^{3 - \alpha_e} \Gamma(0.5 * (3 - alpha_e) {\rm F1}(0.5 (3 - alphae), 3/2, -0.25 (k_1 r)^2))`

    Verified against mathematica numerical integration.

    '''

    cdef double nu, kr

    nu = 3 - alphae
    kr = k1 * r

    return 0.5 * kr**nu * gamma(0.5 * nu) * hyp1f1(0.5 * nu, 1.5, -0.25 * kr**2)


def Int4_inf(double k1, double alphae):
    '''
    Analytical form for Eq. B9

    Use for r >> 1 / k1.

    Valid for alphae > 1, alphae < 3.
    '''

    return gamma(2 - alphae) * sin(alphae * pi * 0.5)

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
    :math:`\frac{1}{2 \pi z_0^2 \theta_0^2} \exp^{-R^2 / 2 \theta_0^2 z_0^2}`

    w_b was solved for using Mathematica. My by-hand solution had the same
    asymptotic behaviour, but looked so much worse than the compacter form
    above.

    '''

    cdef double ratio_term

    ratio_term = R / (2 * theta_0 * z_0)

    return 0.5 * np.exp(- ratio_term**2) / (z_0 * theta_0)**2
    # return 0.5 * np.exp(-0.25 * ratio_term**2)


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
    else:
        return 0.0


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

    '''

    cdef double w_eps_a, w_b_a

    # See form above. Equal to 2 * sigma_z for "theta_0"
    w_eps_a = gaussian_beam(z, 2 * sigma_z)

    w_b_a = gaussian_autocorr(R, z_0, theta0)

    return w_eps_a * w_b_a
