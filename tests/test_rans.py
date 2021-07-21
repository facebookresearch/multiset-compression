"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from multiset_codec import rans

import numpy as np

rng = np.random.RandomState(0)
num_lower_bits = int(np.log2(rans.rans_l)) + 1


def test_rans():
    '''
    Tests identity of encoding and decoding with rANS given some starts
    and freqs ndarrays
    '''

    shape = (8, 7)
    precision = 1337
    n_data = 1000

    x = rans.base_message(shape)
    starts = rng.randint(0, 256, size=(n_data,) + shape).astype("uint64")
    freqs = (rng.randint(1, 256, size=(n_data,) + shape).astype("uint64")
             % (256 - starts))
    freqs[freqs == 0] = 1
    assert np.all(starts + freqs <= 256)

    # Encode
    for start, freq in zip(starts, freqs):
        x = rans.encode(x, start, freq, precision)
    coded_arr = rans.flatten(x)
    assert coded_arr.dtype == np.uint32

    # Decode
    x = rans.unflatten(coded_arr, shape)
    for start, freq in reversed(list(zip(starts, freqs))):
        cf, pop = rans.decode(x, precision)
        assert np.all(start <= cf) and np.all(cf < start + freq)
        x = pop(start, freq)
    assert np.all(x[0] == rans.base_message(shape)[0])


def test_flatten_unflatten():
    '''
    Tests identity of composing flatten and unflatten
    '''
    n = 100
    shape = (7, 3)
    prec = 1337
    state = rans.base_message(shape)
    some_bits = rng.randint(prec, size=(n,) + shape).astype(np.uint64)
    freqs = np.ones(shape, dtype="uint64")
    for b in some_bits:
        state = rans.encode(state, b, freqs, prec)
    flat = rans.flatten(state)
    assert flat.dtype is np.dtype("uint32")
    state_ = rans.unflatten(flat, shape)
    flat_ = rans.flatten(state_)
    assert np.all(flat == flat_)
    assert np.all(state[0] == state_[0])


def test_base_message():
    '''
    Tests if base message is correctly initialized
    '''
    def popcount(head):
        '''
        Counts number of set bits (i.e. 1's) in the lower bits
        of the rANS state head usually 32 bits).
        '''
        return sum([((head >> i) % 2).sum()
                    for i in range(num_lower_bits)])

    # Each element of the head should be initialized to
    # |00000000000000.......00|1000......00|
    # |<- (rans_h - rans_l) ->|<- rans_l ->|
    head, _ = rans.base_message(1_000)
    assert popcount(head) == 1_000

    # Initialize a head with some randomness in the (rans_l-1) lower bits
    head_rnd, _ = rans.base_message(100_000, randomize=True)

    # Each element of the head should be initialized to
    # |00000000000000.......00|1|....random bits...|
    # |<- (rans_h - rans_l) ->| |<- (rans_l - 1) ->|
    assert (head_rnd >> (num_lower_bits - 1) == 1).all()

    # Fraction of random 1's should be close to 50%
    num_bits = num_lower_bits*100_000
    assert 0.48 < popcount(head_rnd)/num_bits < 0.52
