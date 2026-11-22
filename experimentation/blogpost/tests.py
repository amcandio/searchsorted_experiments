from impl import IMPLEMENTATIONS
import numpy as np


def run_case(a, xs):
    ref = np.searchsorted(a, xs, side="left")
    for name, fn in IMPLEMENTATIONS.items():
        out = fn(a, xs)
        assert np.array_equal(out, ref), f"{name} failed for a={a}, xs={xs}. Expected value: {ref} but got {out}"


def test_all():
    # basic
    a = np.array([1, 2, 4, 4, 5])
    xs = np.array([0, 1, 3, 4, 6])
    run_case(a, xs)

    # duplicates
    a = np.array([1, 1, 1, 2, 2, 3, 5, 5, 5])
    xs = np.array([1, 2, 4, 5])
    run_case(a, xs)

    # random
    rng = np.random.default_rng(42)
    for _ in range(100):
        a = np.sort(rng.integers(0, 100, size=50))
        xs = rng.integers(-10, 110, size=100)
        run_case(a, xs)

    print("All implementations match numpy.searchsorted")

if __name__ == "__main__":
    test_all()