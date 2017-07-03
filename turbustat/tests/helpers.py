from __future__ import print_function, absolute_import, division

from astropy import units as u

from numpy.testing import assert_allclose as assert_allclose_numpy, assert_array_equal


def assert_allclose(q1, q2, **kwargs):
    """
    Quantity-safe version of Numpy's assert_allclose

    Copyright (c) 2014, spectral-cube developers
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """
    if isinstance(q1, u.Quantity) and isinstance(q2, u.Quantity):
        assert_allclose_numpy(q1.to(q2.unit).value, q2.value, **kwargs)
    elif isinstance(q1, u.Quantity):
        assert_allclose_numpy(q1.value, q2, **kwargs)
    elif isinstance(q2, u.Quantity):
        assert_allclose_numpy(q1, q2.value, **kwargs)
    else:
        assert_allclose_numpy(q1, q2, **kwargs)