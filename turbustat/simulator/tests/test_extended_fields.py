
from ..gen_field import make_3dfield, make_extended

import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize(('shape', 'slope'), [(shape, slope) for shape in
                                              [32, 33] for slope in
                                              np.arange(0.0, 5.5, 1.)])
def test_3D_gen_field(shape, slope):
    '''
    Power needs to be conserved between the fft and real versions.
    '''

    cube_fft = make_3dfield(shape, powerlaw=slope, return_fft=True)

    cube = make_3dfield(shape, powerlaw=slope, return_fft=False)

    refft = np.fft.rfftn(cube)

    npt.assert_allclose(refft, cube_fft, rtol=1e-8, atol=5e-9)

    power = np.sum(np.abs(cube_fft)**2) / float(cube_fft.size)**2
    power_cube = np.sum(np.abs(refft)**2) / float(refft.size)**2

    npt.assert_allclose(power, power_cube, rtol=1e-8)

    # Std of cube should match the amplitude of 1.
    npt.assert_allclose(1., np.std(cube), rtol=1e-5)


@pytest.mark.parametrize(('shape', 'slope'), [(shape, slope) for shape in
                                              [32, 33] for slope in
                                              np.arange(0.0, 5.5, 1.0)])
def test_2D_gen_field(shape, slope):
    '''
    Power needs to be conserved between the fft and real versions.
    '''

    seed = np.random.randint(0, 2**31 - 1)

    img_fft = make_extended(shape, powerlaw=slope, return_fft=True,
                            randomseed=seed, full_fft=False)

    img = make_extended(shape, powerlaw=slope, return_fft=False,
                        randomseed=seed)

    refft = np.fft.rfft2(img)

    npt.assert_allclose(refft, img_fft, rtol=1e-8, atol=5e-9)

    power = np.sum(np.abs(img_fft)**2) / float(img_fft.size)**2
    power_img = np.sum(np.abs(refft)**2) / float(refft.size)**2

    npt.assert_allclose(power, power_img, rtol=1e-8)
