"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from utils import calculate_state_bits, cache
from multiset_codec import codecs, msbst
from multiset_codec.msbst import build_multiset
from scipy.stats import multinomial, dirichlet
from time import time

import numpy as np
import pandas as pd
import craystack as cs


def run_single_experiment(seq_length, alphabet_size, seed):
    np.random.seed(seed)

    # Sample skewed source from a Dirichlet distribution.
    alphabet = np.arange(alphabet_size)
    source_probs = dirichlet.rvs(alphabet+1).flatten()

    # Create a subset of the alphabet, called alphabet_seen, that contains
    # 512 unique symbols. Only symbols from alphabet_seen will appear in the
    # multiset. This is required to show that the complexity will not scale
    # with alphabet size.
    alphabet_seen = np.random.choice(
            alphabet, size=512, p=source_probs, replace=False)
    source_probs_seen = source_probs[alphabet_seen]
    source_probs_seen /= source_probs_seen.sum()

    # Sample a random sequence, but guarantee exactly 512 unique symbols.
    # To do this, we start with alphabet_seen, and append seq_length-512
    # symbols sampled from alphabet_seen.
    sequence = np.r_[alphabet_seen, np.random.choice(
            alphabet_seen, size=seq_length-512, p=source_probs_seen)]

    # The symbols will be encoded to the ANS state with a codec
    # that has the exact source probabilities. Note that source_probs
    # has size len(alphabet) > len(alphabet_seen) = 512, hence it scales
    # with alphabet size. However, the codec implementation is efficient,
    # and the dependency does not manifest in these experiments.
    symbol_codec = codecs.Categorical(source_probs, prec=27)

    # Start timing, to estimate compute time
    time_start = time()

    # Populate multiset via successive applications of the insert operation.
    multiset = build_multiset(sequence)

    # Initialize multiset codec
    multiset_codec = codecs.Multiset(symbol_codec)

    # Initialize ANS state
    ans_state = cs.base_message(shape=(1,))

    # Encode multiset
    (ans_state,) = multiset_codec.encode(ans_state, multiset)

    # Calculate the size of the compressed state in bits
    # This is the number of bits used to compress the images as a multiset.
    compressed_length_multiset = calculate_state_bits(ans_state)

    # Decode multiset
    multiset_size = seq_length
    ans_state, multiset_decoded = \
            multiset_codec.decode(ans_state, multiset_size)

    # Check elapsed time
    duration = time() - time_start

    # Check if decoded correctly
    assert msbst.check_multiset_equality(multiset, multiset_decoded)

    # Calculate information content of the multiset
    sequence_seen, counts_seen = np.unique(sequence, return_counts=True)
    counts_alphabet = np.zeros(alphabet_size)
    counts_alphabet[sequence_seen] = counts_seen
    multiset_info_content = \
            -multinomial(seq_length, source_probs).logpmf(counts_alphabet)
    multiset_info_content /= np.log(2)  # convert from nats to bits

    return {
        'compressed_length_multiset': compressed_length_multiset,
        'multiset_info_content': multiset_info_content,
        'alphabet_size': alphabet_size,
        'seq_length': seq_length,
        'duration': duration,
    }


def run_all_experiments(seed):
    '''Runs all experiments. This function is used by plots.py'''

    # Run experiments
    metrics = [
        cache(run_single_experiment)(seq_length, alphabet_size, seed)
        for seq_length in np.logspace(9, 12, 10, base=2, dtype=int)
        for alphabet_size in 2**np.arange(10, 18)
        for seed in np.arange(20)
    ]

    # Save results (run plot.py to see plotted results)
    def lower(s): return s.quantile(0.05)
    def upper(s): return s.quantile(0.95)
    def avg(s): return s.mean()

    return (pd.DataFrame(metrics)
              .groupby(['seq_length', 'alphabet_size'])
              .agg([avg, lower, upper]))
