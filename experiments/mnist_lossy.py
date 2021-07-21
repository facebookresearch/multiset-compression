"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from joblib import Parallel, delayed
from multiset_codec import codecs
from multiset_codec.msbst import to_sequence, build_multiset
from utils import (
        calculate_state_bits,
        load_mnist,
        compress_image_lossy,
        log2_multinomial_coeff,
        cache
)

import pandas as pd
import craystack as cs
import numpy as np
import os


def run_single_experiment(seq_length, seed):
    np.random.seed(seed)

    # Load MNIST test set and convert each image to a bytearray symbol
    # (i.e. a variable-length sequence of bytes), representing the output
    # of the lossy codec (WebP).
    sequence = [compress_image_lossy(image, quality=0, method='WebP')
                for image in load_mnist()[:seq_length]]
    assert len(sequence) == seq_length

    # The ANS state is a compound data-structure composed of a head and tail.
    # It allows performing encode and decode operations in parallel, as long
    # as the head shape is larger than the input symbol shape. The symbols
    # are variable-length sequences of bytes, but the image size (784) is an
    # upper bound on the sequence length, so we initialize the ANS head shape
    # to 784, to guarantee that the symbol codec can always encode/decode a
    # symbol (i.e. the variable-length sequence of bytes) in one operation.
    ans_state = cs.base_message(784)
    symbol_codec = codecs.ByteArray(784)

    # Populate multiset via successive applications of the insert operation.
    multiset = build_multiset(sequence)

    # Initialize multiset codec
    multiset_codec = codecs.Multiset(symbol_codec)

    # Encode multiset
    (ans_state,) = multiset_codec.encode(ans_state, multiset)

    # Calculate the size of the compressed state in bits
    # This is the number of bits used to compress the images as a multiset.
    compressed_length_multiset = calculate_state_bits(ans_state)

    # Decode multiset
    multiset_size = seq_length
    ans_state, multiset_decoded = \
            multiset_codec.decode(ans_state, multiset_size)

    # Check if decoded correctly
    assert (np.sort(to_sequence(multiset))
            == np.sort(to_sequence(multiset_decoded))).all()

    # Calculate number of bits used to compress images as an ordered sequence.
    ans_state = cs.base_message(784)
    (ans_state,) = codecs.Sequence(symbol_codec).encode(ans_state, sequence)
    compressed_length_sequence = calculate_state_bits(ans_state)

    # Calculate saved bits and theoretical limit
    # (i.e. information content of uniform permutations)
    saved_bits = compressed_length_sequence - compressed_length_multiset
    saved_bits_limit = \
            log2_multinomial_coeff(np.unique(sequence, return_counts=True)[1])

    return {
        'seq_length': seq_length,
        'saved_bits': saved_bits,
        'saved_bits_limit': saved_bits_limit,
        'compressed_length_multiset': compressed_length_multiset,
        'compressed_length_sequence': compressed_length_sequence,
    }


def run_all_experiments(seed):
    '''Runs experiments in parallel. This function is used by plots.py'''

    # Run experiments in parallel
    metrics = Parallel(n_jobs=os.cpu_count()-1)(
            delayed(cache(run_single_experiment))(seq_length, seed)
            for seq_length in [1, 5, 15, 20, 50, 100, 1000, 5000, 10000])

    # Save results (run plot.py to see plotted results)
    df = pd.DataFrame(metrics).set_index('seq_length')
    df['pct_saved_bits'] = \
        100*(1 - df.compressed_length_multiset/df.compressed_length_sequence)
    df['pct_saved_bits_limit'] = \
            100*df.saved_bits_limit/df.compressed_length_sequence

    return df
