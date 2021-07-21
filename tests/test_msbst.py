"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from multiset_codec.msbst import (
        insert_then_forward_lookup,
        reverse_lookup_then_remove,
        to_sequence,
        forward_lookup,
        reverse_lookup,
        build_multiset
)


def test_insert_then_forward_lookup():
    '''
    Incrementally test insert_then_forward_lookup function, starting from an
    empty multiset.
    '''

    multiset = ()
    leaf = (), ()

    multiset, (start, freq) = insert_then_forward_lookup(multiset, 'c')
    assert multiset == (1, 'c', *leaf)
    assert (start, freq) == (0, 1)

    multiset, (start, freq) = insert_then_forward_lookup(multiset, 'a')
    assert multiset == (2, 'c', (1, 'a', *leaf), ())
    assert (start, freq) == (0, 1)

    multiset, (start, freq) = insert_then_forward_lookup(multiset, 'a')
    assert multiset == (3, 'c', (2, 'a', *leaf), ())
    assert (start, freq) == (0, 2)

    multiset, (start, freq) = insert_then_forward_lookup(multiset, 'b')
    assert multiset ==  (4, 'c',
                            (3, 'a',
                                (),
                                (1, 'b', *leaf)),
                            ())
    assert (start, freq) == (2, 1)

    multiset, (start, freq) = insert_then_forward_lookup(multiset, 'c')
    assert multiset ==  (5, 'c',
                            (3, 'a',
                                (),
                                (1, 'b', *leaf)),
                            ())
    assert (start, freq) == (3, 2)

    multiset, (start, freq) = insert_then_forward_lookup(multiset, 'e')
    assert multiset ==  (6, 'c',
                            (3, 'a',
                                (),
                                (1, 'b', *leaf)),
                            (1, 'e', *leaf))
    assert (start, freq) == (5, 1)

    multiset, (start, freq) = insert_then_forward_lookup(multiset, 'd')
    assert multiset ==  (7, 'c',
                            (3, 'a',
                                (),
                                (1, 'b', *leaf)),
                            (2, 'e',
                                (1, 'd', *leaf),
                                ()))
    assert (start, freq) == (5, 1)

    multiset, (start, freq) = insert_then_forward_lookup(multiset, 'f')
    assert multiset ==  (8, 'c',
                            (3, 'a',
                                (),
                                (1, 'b', *leaf)),
                            (3, 'e',
                                (1, 'd', *leaf),
                                (1, 'f', *leaf)))
    assert (start, freq) == (7, 1)


def test_reverse_lookup_then_remove():
    '''
    Incrementally test reverse_lookup_then_remove function, starting from
    the last multiset in test_insert_then_forward_lookup.
    '''
    leaf = (), ()
    multiset = (8, 'c',
                   (3, 'a',
                       (),
                       (1, 'b', *leaf)),
                   (3, 'e',
                       (1, 'd', *leaf),
                       (1, 'f', *leaf)))

    # 0 1 2 3 4 5 6 7
    #  a |b| c |d|e|f
    multiset, (start, freq), x = reverse_lookup_then_remove(multiset, 3)
    assert x == 'c'
    assert (start, freq) == (3, 2)
    assert multiset == (7, 'c',
                           (3, 'a',
                               (),
                               (1, 'b', *leaf)),
                           (3, 'e',
                               (1, 'd', *leaf),
                               (1, 'f', *leaf)))

    # 0 1 2 3 4 5 6
    #  a |b|c|d|e|f
    multiset, (start, freq), x = reverse_lookup_then_remove(multiset, 1)
    assert x == 'a'
    assert (start, freq) == (0, 2)
    assert multiset == (6, 'c',
                           (2, 'a',
                               (),
                               (1, 'b', *leaf)),
                           (3, 'e',
                               (1, 'd', *leaf),
                               (1, 'f', *leaf)))

    # 0 1 2 3 4 5
    # a|b|c|d|e|f
    multiset, (start, freq), x = reverse_lookup_then_remove(multiset, 3)
    assert x == 'd'
    assert (start, freq) == (3, 1)
    assert multiset == (5, 'c',
                           (2, 'a',
                               (),
                               (1, 'b', *leaf)),
                           (2, 'e',
                               (),
                               (1, 'f', *leaf)))

    # 0 1 2 3 4
    # a|b|c|e|f
    multiset, (start, freq), x = reverse_lookup_then_remove(multiset, 0)
    assert x == 'a'
    assert (start, freq) == (0, 1)
    assert multiset == (4, 'c',
                           (1, 'a',
                               (),
                               (1, 'b', *leaf)),
                           (2, 'e',
                               (),
                               (1, 'f', *leaf)))

    # 0 1 2 3
    # b|c|e|f
    multiset, (start, freq), x = reverse_lookup_then_remove(multiset, 2)
    assert x == 'e'
    assert (start, freq) == (2, 1)
    assert multiset == (3, 'c',
                           (1, 'a',
                               (),
                               (1, 'b', *leaf)),
                           (1, 'e',
                               (),
                               (1, 'f', *leaf)))

    # 0 1 2
    # b|c|f
    multiset, (start, freq), x = reverse_lookup_then_remove(multiset, 1)
    assert x == 'c'
    assert (start, freq) == (1, 1)
    assert multiset == (2, 'c',
                           (1, 'a',
                               (),
                               (1, 'b', *leaf)),
                           (1, 'e',
                               (),
                               (1, 'f', *leaf)))

    # 0 1
    # b|f
    multiset, (start, freq), x = reverse_lookup_then_remove(multiset, 0)
    assert x == 'b'
    assert (start, freq) == (0, 1)
    assert multiset == (1, 'c',
                           (),
                           (1, 'e',
                               (),
                               (1, 'f', *leaf)))

    # 0
    # f
    multiset, (start, freq), x = reverse_lookup_then_remove(multiset, 0)
    assert x == 'f'
    assert (start, freq) == (0, 1)
    assert multiset == ()


def test_to_sequence():
    xs = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'd']
    assert to_sequence(build_multiset(xs)) == sorted(xs)


def test_forward_lookup():
    xs = build_multiset(4 * ['a']
                      + 3 * ['b']
                      + 1 * ['d'])
    assert forward_lookup(xs, 'a') == (0, 4)
    assert forward_lookup(xs, 'b') == (4, 3)
    assert forward_lookup(xs, 'd') == (7, 1)


def test_reverse_lookup():
    xs = build_multiset(4 * ['a']
                      + 3 * ['b']
                      + 1 * ['d'])
    assert reverse_lookup(xs, 0) == ((0, 4), 'a')
    assert reverse_lookup(xs, 1) == ((0, 4), 'a')
    assert reverse_lookup(xs, 2) == ((0, 4), 'a')
    assert reverse_lookup(xs, 3) == ((0, 4), 'a')
    assert reverse_lookup(xs, 4) == ((4, 3), 'b')
    assert reverse_lookup(xs, 5) == ((4, 3), 'b')
    assert reverse_lookup(xs, 6) == ((4, 3), 'b')
    assert reverse_lookup(xs, 7) == ((7, 1), 'd')
