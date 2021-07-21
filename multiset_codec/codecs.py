"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

-------------------------------------------------------------------

All codecs have two methods, with corresponding signatures

    encode: (ans_state, symbol, *context) -> (ans_state, *context)
    decode: (ans_state, *context) -> (ans_state, symbol, *context)

Note that, since context is passed via unpacking (i.e. *context),
then it is essentially optional. However, the return of encode will
be at least (ans_state,). For more detail regarding codecs,
see github.com/j-towns/craystack
"""

from collections import namedtuple
from multiset_codec.msbst import (
    reverse_lookup_then_remove,
    insert_then_forward_lookup
)
from multiset_codec import rans

import numpy as np
import craystack as cs

from typing import Callable

Codec = namedtuple('Codec', ['encode', 'decode'])
ViewFunc = Callable[[np.ndarray], np.ndarray]


def substack(codec: Codec, view_fun: ViewFunc) -> Codec:
    '''
    Apply a codec on a subset of a ans_state head.
    view_fun should be a function: head -> subhead, for example
    view_fun = lambda head: head[0]
    to run the codec on only the first element of the head
    '''
    def encode(ans_state, symbol, *context):
        head, tail = ans_state
        subhead, update = cs.util.view_update(head, view_fun)
        (subhead, tail), *context = \
                codec.encode((subhead, tail), symbol, *context)
        return ((update(subhead), tail), *context)

    def decode(ans_state, *context):
        head, tail = ans_state
        subhead, update = cs.util.view_update(head, view_fun)
        (subhead, tail), symbol, *context = \
                codec.decode((subhead, tail), *context)
        return ((update(subhead), tail), symbol, *context)

    return Codec(encode, decode)


def Multiset(symbol_codec: Codec) -> Codec:
    '''
    Encodes a multiset using bits-back coding.

    Symbols are sampled from the multiset with SamplingWithoutReplacement,
    and encoded sequentially with symbol_codec.
    '''
    swor_codec = SamplingWithoutReplacement()

    def encode(ans_state, multiset):
        while multiset:
            # 1) Sample, without replacement, a symbol using ANS decode.
            ans_state, symbol, multiset = \
                    swor_codec.decode(ans_state, multiset)

            # 2) Encode the selected symbol onto the same ANS state.
            (ans_state,) = symbol_codec.encode(ans_state, symbol)
        return (ans_state,)

    def decode(ans_state, multiset_size):
        multiset = ()
        for _ in range(multiset_size):
            # Decode symbol on top of stack (reverses step 2)
            ans_state, symbol = symbol_codec.decode(ans_state)

            # Encode bits used to sample symbol (reverses step 1)
            # This is the bits-back step!
            ans_state, multiset = \
                    swor_codec.encode(ans_state, symbol, multiset)
        return ans_state, multiset

    return Codec(encode, decode)


def SamplingWithoutReplacement() -> Codec:
    '''
    Encodes and decodes onto the ANS state using the empirical
    distribution of symbols in the multiset.

    Before an encode, the symbol to be encoded is inserted into the multiset.
    After a decode, the decoded symbol is removed from the multiset. Therefore,
    a decode performs sampling without replacement, while encode inverts it.

    The context is the multiset, i.e. *context = multiset
    '''
    def encode(ans_state, symbol, multiset):
        multiset, (start, freq) = insert_then_forward_lookup(multiset, symbol)
        multiset_size = multiset[0]
        ans_state = rans.encode(ans_state, start, freq, multiset_size)
        return ans_state, multiset

    def decode(ans_state, multiset):
        multiset_size = multiset[0]
        cdf_value, decode_ = rans.decode(ans_state, multiset_size)
        multiset, (start, freq), symbol = \
                reverse_lookup_then_remove(multiset, cdf_value[0])
        ans_state = decode_(start, freq)
        return ans_state, symbol, multiset

    return substack(Codec(encode, decode), lambda head: head[:1])


def Uniform(prec: int) -> Codec:
    '''
    Encodes and decodes onto the ANS state using a uniform
    distribution in the interval [0, prec).
    '''
    def encode(ans_state, symbol):
        ans_state = rans.encode(ans_state, symbol, 1, prec)
        return (ans_state,)

    def decode(ans_state):
        symbol, decode_ = rans.decode(ans_state, prec)
        ans_state = decode_(symbol, 1)
        return ans_state, symbol

    return Codec(encode, decode)


def ByteArray(max_array_size: int) -> Codec:
    '''
    Encodes and decodes an array of bytes onto the ANS state.

    First, the bytearray size is encoded using a uniform distribution in
    the interval [0, max_array_size). Then, the bytes are encoded in parallel
    using a uniform distribution in the interval [0, 256).
    '''

    size_codec = substack(Uniform(max_array_size), lambda h: h[:1])
    bytes_codec = lambda size: substack(Uniform(256), lambda h: h[:size])

    def encode(ans_state, bytes_array):
        bytes_ndarray = np.frombuffer(bytes_array, dtype=np.uint8)
        size = len(bytes_array)
        (ans_state,) = bytes_codec(size).encode(ans_state, bytes_ndarray)
        (ans_state,) = size_codec.encode(ans_state, size)
        return (ans_state,)

    def decode(ans_state):
        ans_state, size = size_codec.decode(ans_state)
        ans_state, bytes_ndarray = bytes_codec(size[0]).decode(ans_state)
        bytes_array = bytes_ndarray.astype(np.uint8).tobytes()
        return ans_state, bytes_array

    return Codec(encode, decode)


def Categorical(probs: np.ndarray, prec: int) -> Codec:
    '''
    Encodes and decodes according to distribution probs at precision prec.
    '''
    _encode, decode = cs.Categorical(probs, prec)

    def encode(ans_state, symbol):
        ans_state = _encode(ans_state, symbol)
        return (ans_state,)

    return Codec(encode, decode)


def Sequence(symbol_codec: Codec) -> Codec:
    '''
    Encodes a sequence by sequentially encoding symbols with symbol_codec.
    '''
    def encode(ans_state, sequence, *context):
        for symbol in sequence:
            ans_state, *context = \
                    symbol_codec.encode(ans_state, symbol, *context)
        return (ans_state, *context)

    def decode(ans_state, seq_length, *context):
        sequence = seq_length*[None]
        for i in reversed(range(seq_length)):
            ans_state, sequence[i], *context = \
                    symbol_codec.decode(ans_state, *context)
        return (ans_state, sequence, *context)

    return Codec(encode, decode)


def VariableLengthSequence(symbol_codec: Codec, max_seq_length: int) -> Codec:
    '''
    Encodes a variable-length sequence by sequentially encoding symbols with the
    symbol_codec, followed by encoding the size of the sequence.
    '''
    sequence_codec = Sequence(symbol_codec)
    seq_length_codec = substack(Uniform(max_seq_length+1), lambda h: h[:1])

    def encode(ans_state, sequence, *context):
        (ans_state, *context) = \
                sequence_codec.encode(ans_state, reversed(sequence), *context)
        (ans_state,) = seq_length_codec.encode(ans_state, len(sequence))
        return (ans_state, *context)

    def decode(ans_state, *context):
        (ans_state, seq_length) = seq_length_codec.decode(ans_state)
        (ans_state, sequence, *context) = \
                sequence_codec.decode(ans_state, seq_length[0], *context)
        return (ans_state, sequence, *context)

    return Codec(encode, decode)
