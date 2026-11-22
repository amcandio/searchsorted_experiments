import numpy as np
import time

from impl import IMPLEMENTATIONS

REPEATS = 50

# Generate logarithmically spaced sizes including powers of 10
def generate_sizes(min_i=8, max_i=29):
    """
    Generate sizes as 2**i, 1.25*2**i, 1.5*2**i, 1.75*2**i
    for i in range(min_i, max_i + 1)
    """
    sub = 8
    multipliers = [1 + i / sub for i in range(sub)]
    sizes = []
    for i in range(min_i, max_i + 1):
        base = 2 ** i
        for m in multipliers:
            sizes.append(int(m * base))
    return sizes


SIZES = generate_sizes()


def benchmark(fn, a, xs, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(a, xs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.min(times)

NUM_QUERIES = 10_000  # fixed number of search queries

def run_benchmarks():
    rng = np.random.default_rng(123)

    print("n,method,time")  # CSV header

    for n in SIZES:
        # sorted array
        a = np.sort(rng.integers(0, n, size=n, dtype=np.int32))

        # fixed number of queries
        xs = rng.integers(0, n, size=NUM_QUERIES, dtype=np.int32)

        for name, fn in IMPLEMENTATIONS.items():
            t = 10**9 * benchmark(fn, a, xs, REPEATS) / NUM_QUERIES
            print(f"{n},{name},{t:.6f}")


if __name__ == "__main__":
    run_benchmarks()

