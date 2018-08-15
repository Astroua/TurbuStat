
cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport sqrt

cdef double pi = np.pi
cdef double SQRT_2PI = sqrt(2 * pi)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def generate_spectrum(np.ndarray[np.float64_t, ndim=1] vel,
                      np.ndarray[np.float64_t, ndim=1] dens,
                      np.ndarray[np.float64_t, ndim=1] vel_edges,
                      np.ndarray[np.float64_t, ndim=1] v_cents,
                      np.ndarray[np.float64_t, ndim=1] dvdz,
                      double v_therm_sq,
                      double pix_scale):
    '''
    Generate an optically-thin spectrum at one line-of-sight given the velocity
    and density fields.

    Quantities MUST be given in CGS units:

    * vel - cm/s
    * dens - cm^-3
    * vel_edges - cm/s
    * v_cents - cm/s
    * dvdz - 1 / s
    * v_therm_sq - (cm/s)^2
    * pix_scale - cm

    '''

    cdef int i, jj, ni
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] contribs
    cdef np.ndarray[np.float64_t, ndim=1] spectrum
    cdef np.ndarray[np.float64_t, ndim=1] sigma
    cdef double term1
    cdef np.ndarray[np.float64_t, ndim=1] term2

    spectrum = np.zeros_like(vel_edges[:-1])

    for i, (vel_l, vel_h) in enumerate(zip(vel_edges[:-1], vel_edges[1:])):

        # Find locations of all contributions at this velocity
        contribs = (vel > vel_l) & (vel < vel_h)

        if not contribs.any():
            continue

        sigma = np.sqrt((dvdz[contribs] * pix_scale)**2 + v_therm_sq)

        for jj, ni in enumerate(np.where(contribs)[0]):

            # Column density normalized by Gaussian integral
            # Output in units of u.cm**-2 / (u.cm / u.s)
            term1 = (dens[ni] * pix_scale) / (SQRT_2PI * sigma[jj])

            term2 = ((v_cents - vel[ni])**2 /
                     (2 * sigma[jj]**2))

            spectrum += (term1 * np.exp(-term2))

    return spectrum
