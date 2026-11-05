import random

from medret.eval.lcs import LCSMatcher, lcs_length


def lcs_dp(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def test_matches_dp_on_random_sequences():
    rng = random.Random(0)
    for _ in range(500):
        a = [rng.randrange(6) for _ in range(rng.randrange(0, 40))]
        b = [rng.randrange(6) for _ in range(rng.randrange(0, 40))]
        assert lcs_length(a, b) == lcs_dp(a, b)


def test_long_sequences():
    rng = random.Random(1)
    a = [rng.randrange(50) for _ in range(700)]
    b = [rng.randrange(50) for _ in range(300)]
    assert LCSMatcher(a).lcs(b) == lcs_dp(a, b)


def test_edge_cases():
    assert lcs_length([], [1, 2]) == 0
    assert lcs_length("abc", "abc") == 3
    assert lcs_length("abc", "xyz") == 0
    assert lcs_length("a b c".split(), "a x b y c".split()) == 3  # subsequence, gaps allowed
