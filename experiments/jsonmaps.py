"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from joblib import Parallel, delayed
from multiset_codec import codecs
from multiset_codec.msbst import (
        to_sequence,
        build_multiset,
        check_multiset_equality
)
from utils import (
        calculate_state_bits,
        load_corpus,
        cache
)
from json import loads as parse_json

import pandas as pd
import craystack as cs
import numpy as np
import os


def run_single_experiment(seq_length, seed):
    np.random.seed(seed)

    # Load JSON maps and convert each to a list of 2D tuples of UTF-8 bytes,
    # representing the (key, value) pairs. All key-value pairs are
    # cast to strings for simplicity. The list of tuples is sorted,
    # to impose an ordering between JSONs that can be used by bits-back.
    path = 'data/github-users.jsonl'
    jsons_as_lists_of_tuples = [
            sorted(parse_json(json).items())
            for json in load_corpus(path)[:seq_length]
        ]
    sequence = [[(str(k).encode('utf-8'), str(v).encode('utf-8'))
                  for k, v in json]
                 for json in jsons_as_lists_of_tuples]

    # We assume all JSON maps are unique and have exactly 17 key-value pairs,
    # although this is trivially generalizable by encoding the number of
    # key-value pairs as the last step of encoding.
    assert all(len(json) == 17 for json in sequence)
    assert len(sequence) == seq_length
    assert len({frozenset(json) for json in sequence}) == seq_length

    # Populate outer and inner multisets via successive applications of the
    # insert operation. The inner multisets, each representing some JSON map,
    # are the elements of the nodes of the outer multiset.
    multiset_outer = \
            build_multiset([build_multiset(json) for json in sequence])

    # Initialize codec for sampling without replacement
    swor_codec = codecs.SamplingWithoutReplacement()

    # The ANS state is a compound data-structure composed of a head and tail.
    # It allows performing encode and decode operations in parallel, as long
    # as the head shape is larger than the input symbol shape. The symbols
    # are UTF-8 bytearrays (i.e. keys and values), which vary in length.
    # We initialize the ANS head shape to 1000, which means that neither
    # the keys or values can exceed 1000 UTF-8 bytes in size.
    ans_state = cs.base_message(1000)
    utf8_codec = codecs.ByteArray(1000)

    # Encode
    while multiset_outer:
        # 1) Sample, without replacement, an inner multiset using ANS decode
        ans_state, multiset_inner, multiset_outer = \
                swor_codec.decode(ans_state, multiset_outer)

        while multiset_inner:
            # 2) Sample, without replacement, a key-value pair using ANS decode
            ans_state, (key, value), multiset_inner = \
                    swor_codec.decode(ans_state, multiset_inner)

            # 3) Encode the selected key-value pair onto the same ANS state.
            (ans_state,) = utf8_codec.encode(ans_state, key)
            (ans_state,) = utf8_codec.encode(ans_state, value)

    # Calculate the size of the compressed state in bits
    # This is the number of bits used to compress the JSON maps as a multiset.
    compressed_length_multiset = calculate_state_bits(ans_state)

    # Decode
    multiset_outer_decoded = ()
    for _ in range(seq_length):
        multiset_inner_decoded = ()
        for _ in range(17):
            # Decode key-value pair on top of stack (reverses step 3)
            ans_state, value = utf8_codec.decode(ans_state)
            ans_state, key = utf8_codec.decode(ans_state)

            # Encode bits used to sample key-value pair (reverses step 2)
            # This is the first bits-back step!
            ans_state, multiset_inner_decoded = \
                    swor_codec\
                    .encode(ans_state, (key, value), multiset_inner_decoded)

        # Encode bits used to sample inner multiset (reverses step 1)
        # Sorting the inner multiset is needed to properly account for
        # the ordering between JSON items defined on line 31. Alternatively,
        # we could create a class for the inner multisets to override the
        # comparison operators __le__ and __ge__.
        # This is the second bits-back step!
        json = build_multiset(sorted(to_sequence(multiset_inner_decoded)))
        ans_state, multiset_outer_decoded = \
                swor_codec.encode(ans_state, json, multiset_outer_decoded)

    # Check if decoded correctly
    # Rebuild the original multiset, as it was consumed during encoding
    multiset_outer = \
            build_multiset([build_multiset(json) for json in sequence])
    assert check_multiset_equality(multiset_outer, multiset_outer_decoded)

    # Calculate number of bits needed to compress JSON maps as an ordered
    # sequence. Since all JSONs are assumed to have exactly 17 key-value pairs
    # then encoding all key-value elements sequentially is sufficient to
    # reconstruct the JSON maps losslessly.
    ans_state = cs.base_message(1000)
    sequence_of_json_elements = \
            [el for json in sequence
                for key_value_pair in json
                for el in key_value_pair]
    (ans_state,) = codecs.Sequence(utf8_codec)\
                         .encode(ans_state, sequence_of_json_elements)
    compressed_length_sequence = calculate_state_bits(ans_state)

    # Calculate saved bits and theoretical limit. Assumes all maps are unique,
    # so that the theoretical limit on savings is
    # log2(seq_length!) + seq_length*log2(17!).
    saved_bits = compressed_length_sequence - compressed_length_multiset
    saved_bits_limit_unnested = \
            np.sum(np.log2(np.arange(seq_length)+1))
    saved_bits_limit = \
            saved_bits_limit_unnested \
            + seq_length*np.sum(np.log2(np.arange(17)+1))

    return {
        'seq_length': seq_length,
        'saved_bits': saved_bits,
        'saved_bits_limit': saved_bits_limit,
        'saved_bits_limit_unnested': saved_bits_limit_unnested,
        'compressed_length_multiset': compressed_length_multiset,
        'compressed_length_sequence': compressed_length_sequence,
    }


def run_all_experiments(seed):
    '''Runs experiments in parallel. This function is used by plots.py'''

    # Run experiments in parallel
    metrics = Parallel(n_jobs=os.cpu_count() - 1)(
            delayed(cache(run_single_experiment))(seq_length, seed)
            for seq_length in [1, 2, 5, 15, 20, 50, 100, 1000, 5000])

    # Save results (run plot.py to see plotted results)
    df = pd.DataFrame(metrics).set_index('seq_length')
    df['pct_saved_bits'] = \
        100*(1 - df.compressed_length_multiset/df.compressed_length_sequence)
    df['pct_saved_bits_limit'] = \
            100*df.saved_bits_limit/df.compressed_length_sequence
    df['pct_saved_bits_limit_unnested'] = \
            100*df.saved_bits_limit_unnested/df.compressed_length_sequence

    return df
