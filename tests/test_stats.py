from llmbench.stats import median_or_none, percentile


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4


def test_percentile_empty_and_single():
    assert percentile([], 90) is None
    assert percentile([5], 90) == 5


def test_median_or_none():
    assert median_or_none([]) is None
    assert median_or_none([1, 3]) == 2.0
