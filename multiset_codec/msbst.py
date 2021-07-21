"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

-------------------------------------------------------------------

The multiset is a binary search tree implemented as a nested tuple tree,
together with the following methods

    insert: (multiset, symbol) -> multiset
    remove: (multiset, symbol) -> multiset

    forward_lookup: (multiset, symbol) -> (start, freq)
    reverse_lookup: (multiset, idx) -> (start, freq), symbol

where start/freq are the cumulative/frequency counts required to perform
ANS encode and decode.

See the README for an example.
"""

from functools import reduce


def insert(multiset, x):
    '''Inserts a symbol x into the multiset'''
    size, y, left, right = multiset or (0, x, (), ())
    if x < y:
        left = insert(left, x)
    elif x > y:
        right = insert(right, x)
    return size + 1, y, left, right


def remove(multiset, x):
    '''Removes a symbol x from the multiset'''
    size, y, left, right = multiset
    if size == 1:
        return ()
    if x < y:
        left = remove(left, x)
    elif x > y:
        right = remove(right, x)
    return size - 1, y, left, right


def forward_lookup(multiset, x):
    '''
    Looks up the cumulative (start) and frequency (freq) counts of symbol x.
    '''
    if not multiset:
        raise ValueError("The symbol {} could not be found.".format(x))
    size, y, left, right = multiset
    if x > y:
        start_right, freq = forward_lookup(right, x)
        start = size - right[0] + start_right
    elif x < y:
        start, freq = forward_lookup(left, x)
    else:
        start = left[0] if left else 0
        freq = size - start - (right[0] if right else 0)
    return start, freq


def reverse_lookup(multiset, idx):
    '''
    Looks up the cumulative (start) and frequency (freq) counts,
    as well as the symbol x, at index idx.
    '''
    size, y, left, right = multiset or (0, (), (), ())
    assert 0 <= idx < size
    y_start = left[0] if left else 0
    y_freq = size - y_start - (right[0] if right else 0)
    if idx < y_start:
        (start, freq), x = reverse_lookup(left, idx)
    elif idx >= y_start + y_freq:
        size_not_right = size - right[0]
        (start, freq), x = reverse_lookup(right, idx - size_not_right)
        start = start + size_not_right
    else:
        x, start, freq = y, y_start, y_freq
    return (start, freq), x


def insert_then_forward_lookup(multiset, x):
    ''' Performs insert followed by forward_lookup, in one-pass.'''
    size, y, left, right = multiset or (0, x, (), ())
    size = size + 1
    if x > y:
        right, (start_right, freq) = insert_then_forward_lookup(right, x)
        start = size - right[0] + start_right
    elif x < y:
        left, (start, freq) = insert_then_forward_lookup(left, x)
    else:
        start = left[0] if left else 0
        freq = size - start - (right[0] if right else 0)
    return (size, y, left, right), (start, freq)


def reverse_lookup_then_remove(multiset, idx):
    ''' Performs reverse_lookup followed by remove, in one-pass.'''
    size, y, left, right = multiset
    y_start = left[0] if left else 0
    y_freq = size - y_start - (right[0] if right else 0)
    if idx < y_start:
        left, (start, freq), x = reverse_lookup_then_remove(left, idx)
    elif idx >= y_start + y_freq:
        size_not_right = size - right[0]
        right, (start, freq), x = \
                reverse_lookup_then_remove(right, idx - size_not_right)
        start = start + size_not_right
    else:
        x, start, freq = y, y_start, y_freq
    size = size - 1
    return (size, y, left, right) if size else (), (start, freq), x


def build_multiset(sequence):
    '''Builds a multiset from the sequence by applying insert sequentially'''
    return tuple(reduce(insert, sequence, ()))


def to_sequence(multiset):
    ''' Flatten a BST, representing a multiset, to a sequence (python list)'''
    flat = []

    def traverse(branch):
        if branch:
            size, y, left, right = branch
            traverse(left)
            freq = size - (left[0] if left else 0) - (right[0] if right else 0)
            flat.extend(freq * [y])
            traverse(right)
    traverse(multiset)
    return flat


def check_multiset_equality(multiset, other_multiset):
    return sorted(to_sequence(multiset)) \
            == sorted(to_sequence(other_multiset))
