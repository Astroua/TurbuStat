# Licensed under an MIT open source license - see LICENSE

import pytest
import numpy as np
from astropy.io.fits.header import Header


from ..io import input_data
from ._testing_data import dataset1, sc1, moment0_hdu1


@pytest.mark.parametrize(('data', 'no_header'),
                         [(sc1, False),
                          (dataset1['cube'], False),
                          (dataset1['cube'][0], True),
                          (dataset1["moment0"], False),
                          (list(dataset1["moment0"]), False),
                          (moment0_hdu1, False)])
def test_input_data(data, no_header):

    output_data = input_data(data, no_header=no_header)

    assert isinstance(output_data[0], np.ndarray)
    if not no_header:
        assert isinstance(output_data[1], Header)
