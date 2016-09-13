
import pytest
import astropy.units as u

from ._testing_data import header
from ..io import find_beam_width


def test_load_beam():

    beam_header = header.copy()
    beam_header["BMAJ"] = 1.0

    beamwidth = find_beam_width(beam_header)

    assert beamwidth == 1.0 * u.deg


@pytest.mark.xfail(raises=ValueError)
def test_load_beam_fail():

    find_beam_width(header)
