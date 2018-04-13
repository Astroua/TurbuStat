# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest
import astropy.units as u

try:
    from radio_beam.beam import NoBeamException
    RADIO_BEAM_INSTALLED = True
except ImportError:
    RADIO_BEAM_INSTALLED = False

from ._testing_data import header
from ..io import find_beam_width, find_beam_properties


def test_load_beam():

    beam_header = header.copy()
    beam_header["BMAJ"] = 1.0

    beamwidth = find_beam_width(beam_header)

    assert beamwidth == 1.0 * u.deg


@pytest.mark.parametrize(('major', 'minor', 'pa'), [(1.0, 0.5, 10),
                                                    (1.0, 'skip', 10),
                                                    (1.0, 0.5, 'skip')])
def test_load_beam_props(major, minor, pa):

    beam_header = header.copy()
    beam_header["BMAJ"] = major
    if minor != 'skip':
        beam_header["BMIN"] = minor
    if pa != 'skip':
        beam_header["BPA"] = pa

    bmaj, bmin, bpa = find_beam_properties(beam_header)

    assert bmaj == major * u.deg

    if minor == 'skip':
        assert bmin == major * u.deg
    else:
        assert bmin == minor * u.deg

    if pa == "skip":
        assert bpa == 0 * u.deg
    else:
        assert bpa == pa * u.deg


# radio-beam no has an exception for when no beam is found.
# @pytest.mark.skipif("not RADIO_BEAM_INSTALLED")
if RADIO_BEAM_INSTALLED:
    @pytest.mark.xfail(raises=NoBeamException)
    def test_load_beam_fail():

        find_beam_width(header)

