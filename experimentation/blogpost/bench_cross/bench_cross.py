import os

# NUM_QUERIES=20000 NUM_THREADS=1 python bench2.py > results_cross/parallel_1_20k.csv;
# NUM_QUERIES=20000 NUM_THREADS=8 python bench2.py > results_cross/parallel_8_3_20k.csv;

# ============================================================
# Benchmark configuration
# ============================================================

NUM_THREADS = int(os.environ.get("NUM_THREADS", 1))

# Must be configured before importing JAX.
os.environ["JAX_PLATFORMS"] = "cpu"

import time

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

import numba as nb

# TensorFlow runtime threading controls must be set before initialization.
if NUM_THREADS:
    tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

import torch

if NUM_THREADS:
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(NUM_THREADS)


REPEATS = 100
NUM_QUERIES = int(os.environ.get("NUM_QUERIES", 10_000))


def generate_sizes(min_i=8, max_i=29):
    sub = 8
    multipliers = [1 + i / sub for i in range(sub)]

    sizes = []

    for i in range(min_i, max_i + 1):
        base = 2**i

        for m in multipliers:
            sizes.append(int(m * base))

    return sizes


SIZES = generate_sizes()


def wait(result):
    """Ensure asynchronous execution has completed."""
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()


def benchmark(fn, a, xs, repeats=REPEATS):
    # Warm up outside the timed region.
    result = fn(a, xs)
    wait(result)

    times = []

    for _ in range(repeats):
        t0 = time.perf_counter()

        result = fn(a, xs)
        wait(result)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times)


def setup_numpy(a, xs):
    return a, xs


def setup_torch(a, xs):
    return (
        torch.from_numpy(a).to("cpu"),
        torch.from_numpy(xs).to("cpu"),
    )


def setup_jax(a, xs):
    cpu = jax.devices("cpu")[0]

    return (
        jax.device_put(a, cpu),
        jax.device_put(xs, cpu),
    )


def setup_tensorflow(a, xs):
    with tf.device("/CPU:0"):
        return (
            tf.convert_to_tensor(a),
            tf.convert_to_tensor(xs),
        )

IMPLEMENTATIONS = {
    "numpy.searchsorted": {
        "fn": np.searchsorted,
        "setup": setup_numpy,
        "version": np.__version__,
    },
    "torch.searchsorted": {
        "fn": torch.searchsorted,
        "setup": setup_torch,
        "version": torch.__version__,
    },
    "jax.numpy.searchsorted": {
        "fn": jnp.searchsorted,
        "setup": setup_jax,
        "version": jax.__version__,
    },
    "tensorflow.searchsorted": {
        "fn": tf.searchsorted,
        "setup": setup_tensorflow,
        "version": tf.__version__,
    },
}


def run_benchmarks():
    rng = np.random.default_rng(123)
    print("n,method,time")

    for n in SIZES:
        a_np = np.sort(
            rng.integers(
                0,
                n,
                size=n,
                dtype=np.int32,
            )
        )

        xs_np = rng.integers(
            0,
            n,
            size=NUM_QUERIES,
            dtype=np.int32,
        )

        expected = np.searchsorted(a_np, xs_np)

        for name, impl in IMPLEMENTATIONS.items():
            # Setup happens outside the timed region.
            a, xs = impl["setup"](a_np, xs_np)

            # Correctness check.
            result = impl["fn"](a, xs)
            wait(result)

            if not np.array_equal(np.asarray(result), expected):
                raise RuntimeError(
                    f"{name} produced incorrect results"
                )

            elapsed = benchmark(
                impl["fn"],
                a,
                xs,
            )

            ns_per_query = (
                1e9 * elapsed / NUM_QUERIES
            )

            method_name = f"{name} ({impl['version']})"
            print(f"{n},{method_name},{ns_per_query:.6f}")


if __name__ == "__main__":
    run_benchmarks()