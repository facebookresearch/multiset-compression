"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from utils import log2_multinomial_coeff

import numpy as np


def test_log2_multinomial_coeff():
    f = log2_multinomial_coeff
    assert f(np.array([1, 1])) == 1
    assert np.isclose(f(np.array([1, 1, 1, 1])), np.log2(4*3*2*1))
    assert np.isclose(f(np.array([1, 2, 2, 1])), np.log2(6*5*4*3*2*1/(2*2)))
