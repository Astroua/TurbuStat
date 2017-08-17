
import numpy as np
from scipy.integrate import tplquad
from astropy import constants as const
import astropy.units as u
from functools import partial
import math

try:
    import vegas
    VEGAS_IMPORT_FLAG = True
except ImportError:
    VEGAS_IMPORT_FLAG = False


from vcs_model_helpers import (Dz, C_eps, F_eps_norm, pencil_beam_slab_z,
                               pencil_beam_gaussian_z,
                               gaussian_beam_slab_z_parallel,
                               gaussian_beam_slab_z_crossing,
                               gaussian_beam_gaussian_z_parallel)


def P1(kv, alphav, alphae, P0, k0, V0, T, b=0,
       beam_type='gaussian', object_type='gaussian', los_type='parallel',
       z0=None, z1=None, sigma_z=None, theta0=None, k1=None,
       integration_type='mc', **integration_kwargs):
    '''
    VCS model.
    '''

    fk2 = f_k(kv, T)**2

    window = partial(window_function, beam_type=beam_type,
                     object_type=object_type, los_type=los_type,
                     sigma_z=sigma_z, z0=z0, z1=z1, theta0=theta0)

    # Set bounds based on the inputs
    if object_type == "gaussian":
        z_bounds = [z0 - 5 * sigma_z, z0 + 5 * sigma_z]
    else:
        # Only consider values within the slab
        z_bounds = [z0, z1]

    if beam_type == "gaussian":
        # becomes highly attenuated near the beam size
        r_bounds = [0, 1.5 * theta0 * z0]
    else:
        # Pencil beam
        raise NotImplementedError("A reasonable bound for R still needs to be"
                                  " implemented!")
        r_bounds = [0, ]

    if alphae == 3:

        def integrand(z, theta, r):

            # Extra factor of r on each for the cylindrical Jacobian
            # No density contribution
            return window(r, z) * \
                math.exp(-0.5 * kv**2 * Dz(r, theta, V0, k0, alphav))  # -
                       # 1.0j * kv * b * z) * r

    elif alphae > 3:

        norm_factor = F_eps_norm(alphae, k0)

        def integrand(z, theta, r):

            # Extra factor of r on each for the cylindrical Jacobian
            # Steep density
            return window(r, z) * \
                C_eps(r, k0, alphae, norm_factor) * \
                math.exp(-0.5 * kv**2 * Dz(r, theta, V0, k0, alphav))  # -
                       # 1.0j * kv * b * z) * r
    else:

        norm_factor = F_eps_norm(alphae, k1)

        def integrand(z, theta, r):

            # Extra factor of r on each for the cylindrical Jacobian
            # Shallow density
            return window(r, z) * \
                C_eps(r, k1, alphae, norm_factor) * \
                math.exp(-0.5 * kv**2 * Dz(r, theta, V0, k0, alphav))  # -
                       # 1.0j * kv * b * z) * r

    if integration_type == "quad":
        # Integration order is from the last to first arguments
        # So z, theta, then r
        value, error = tplquad(integrand, 0, r_bounds[1],
                               lambda r: 0.0, lambda r: 2 * np.pi,
                               lambda r, theta: z_bounds[0],
                               lambda r, theta: z_bounds[1],
                               **integration_kwargs)
    elif integration_type == 'mc':
        if not VEGAS_IMPORT_FLAG:
            raise ImportError("Monte Carlo integration require the vegas "
                              "package.")
        integ = vegas.Integrator([z_bounds,
                                  [0, 2 * np.pi],
                                  r_bounds])

        def wrap_integrand(vals):
            z, r, theta = vals
            return integrand(z, r, theta)

        result = integ(wrap_integrand, **integration_kwargs)
        value = result.mean
        error = np.sqrt(result.var)

    return P0 * fk2 * value, P0 * fk2 * error
    # return P0 * value, P0 * error


k_B = const.k_B.to(u.J / u.K).value
m_p = const.m_p.to(u.kg).value


def f_k(kv, T):
    return np.exp(- 0.5 * k_B * T * kv**2 / m_p)


# Window functions and related


def window_function(r, z, beam_type, object_type, los_type, sigma_z=None,
                    z0=None, z1=None, theta0=None):
    '''
    Return the window auto-correlation function for a variety of cases.
    '''

    beam_types = ['pencil', 'gaussian']
    object_types = ['slab', 'gaussian']
    los_types = ['parallel', 'crossing']

    if beam_type not in beam_types:
        raise ValueError("beam_type must be one of {}".format(beam_types))
    if object_type not in object_types:
        raise ValueError("object_type must be one of "
                         "{}".format(object_types))
    if los_type not in los_types:
        raise ValueError("los_type must be one of {}".format(los_types))

    if beam_type == 'pencil':
        # los_type has no effect hasa
        if object_type == 'slab':
            return pencil_beam_slab_z(z, z0, z1)
        else:
            # Gaussian case
            return pencil_beam_gaussian_z(z, sigma_z)
    else:
        if object_type == 'slab':
            if los_type == 'parallel':
                return gaussian_beam_slab_z_parallel(r, z, z0, z1, theta0)
            else:
                # Crossing
                return gaussian_beam_slab_z_crossing(r, z, z0, z1, theta0)
        else:
            # Gaussian
            if los_type == 'parallel':
                return gaussian_beam_gaussian_z_parallel(r, z, z0, sigma_z,
                                                         theta0)
            else:
                raise NotImplementedError("The gaussian beam, gaussian LOS "
                                          "structure in the parallel limit is"
                                          " not implemented.")
