"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from experiments import mnist_lossy, toy_multisets, jsonmaps


def test_mnist_lossy():
    mnist_lossy.run_single_experiment(seq_length=10, seed=1337)


def test_jsonmaps():
    jsonmaps.run_single_experiment(seq_length=10, seed=1337)


def test_toy_multisets():
    toy_multisets.run_single_experiment(seq_length=512, alphabet_size=512, seed=1337)
