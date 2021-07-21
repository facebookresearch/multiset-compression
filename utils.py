"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from joblib import Memory
from functools import lru_cache
from scipy.special import gammaln

import craystack as cs
import numpy as np
import PIL.Image as pimg
import io

cache = Memory('.cache').cache

@lru_cache()
def load_mnist():
    '''
    Loads the pre-saved MNIST (http://yann.lecun.com/exdb/mnist/)
    test set of 10,000 images.
    '''
    return np.load('data/mnist.npz')['data']

@lru_cache()
def load_corpus(path):
    '''Loads a line-delimited text file as a list of binary strings'''
    with open(path, 'rb') as f:
        return f.readlines()

def calculate_state_bits(ans_state):
    '''
    Calculates the number of bits needed to serialize the ANS state
    to disk.
    '''
    return 8*cs.flatten(ans_state).nbytes

def log2_multinomial_coeff(freqs):
    '''
    Calculates the logarithm of the multinomial coefficient, efficiently.
    This is equivalent to, np.log2(freqs.sum()!/np.prod([f! for f in freqs]))
    '''
    return (gammaln(freqs.sum()+1) - gammaln(freqs+1).sum())/np.log(2)

def compress_image_lossy(image, method, **params):
    '''
    Compresses an image using a lossy compression method.
    The output is a variable-length ndarray of dtype np.uint8 (i.e. bytes)
    '''
    image_bytes = io.BytesIO()
    pimg.fromarray(image).save(image_bytes, format=method, **params)
    return image_bytes.getvalue()
