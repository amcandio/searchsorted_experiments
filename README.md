# searchsorted experiments

Experiments in making batched binary search (`np.searchsorted`) faster. This is a follow-up to the blog post [Making np.searchsorted up to 25× Faster in NumPy 2.5](https://blog.scientific-python.org/numpy/searchsorted/).

## Good reads

- [Binary search, but faster (CuriousCoding)](https://curiouscoding.nl/posts/binsearch/): step-by-step optimization of binary search, including branchless search, prefetching, and memory layouts.
- [Binary Search chapter (Algorithmica)](https://en.algorithmica.org/hpc/data-structures/binary-search/): branchless search, the Eytzinger layout, prefetching and other cache-friendly layouts.
- [You can beat the binary search (Daniel Lemire)](https://lemire.me/blog/2026/04/27/you-can-beat-the-binary-search/): ways to beat plain binary search in practice.
- [Performance comparison: linear search vs binary search (Dirty Hands Coding)](https://dirtyhandscoding.github.io/posts/performance-comparison-linear-search-vs-binary-search.html): where linear scans win over binary search, which motivates the small-range fallback below.

## Ideas to try

- **Fixed batch sizes** instead of scanning the full key array at once. Processing keys in chunks sized to fit in cache (L1/L2) should cut memory traffic for the per-level temporaries and the `k` / `keys` arrays.
- **Fallback to linear scan** once the remaining search range is small enough. A short contiguous scan (or a vectorized compare-and-count over a small window) can beat the last few binary steps, which are branchy and latency-bound.
- **Shared early passes / quaternary search.** The first levels of the search are the same for every key: they all compare against the same few pivots (the same percentiles of the array). We can reduce the search space in bigger steps, e.g. compare against 3 pivots at once (quaternary search) or more (k-ary search), so each pass removes more of the range than a single binary step.
- **Other memory layouts**, such as the Eytzinger layout (implemented below), B-tree / S-tree-style blocked layouts, and van Emde Boas order.

---

## Eytzinger-layout `searchsorted`

Experiments with faster batched binary search in pure Python + NumPy. The layout is described in [Algorithmica's Binary Search chapter](https://en.algorithmica.org/hpc/data-structures/binary-search/), and the blog post above closes by suggesting it as a cache-friendly next step.

The current experiment is an **Eytzinger-layout `searchsorted`** built only from NumPy array operations.
 
Experiments with faster batched binary search in pure Python + NumPy. This is a follow-up to the blog post [Making np.searchsorted up to 25× Faster in NumPy 2.5](https://blog.scientific-python.org/numpy/searchsorted/), which closes by suggesting the Eytzinger layout as a cache-friendly next step. The layout itself is described in [Algorithmica's Binary Search chapter](https://en.algorithmica.org/hpc/data-structures/binary-search/).
 
The current experiment is an **Eytzinger-layout `searchsorted`** built only from NumPy array operations.
 
## What's in here
 
| File | Contents |
|---|---|
| `eytzinger_searchsorted.py` | Eytzinger builders, four search functions, correctness checks and a benchmark (run it as a script) |
 
Requires Python 3 and NumPy.
 
```bash
python eytzinger_searchsorted.py   # runs correctness checks, then the benchmark
```
 
## The idea
 
Take a sorted array and permute it into **Eytzinger (BFS / implicit-heap) order**: node `k` has children `2k` and `2k+1`. The first levels of every search then touch the same few contiguous cache lines instead of jumping around the array.
 
Searching is vectorized across all keys. Every key descends one tree level per step:
 
```python
k = 2 * k + (t[k] < keys)
```
 
All keys take the same number of steps, so there is no `active` mask. At the end, a bit trick (`k >>= bit_length(~k & (k + 1))`) strips the trailing right-moves plus the last left-move, which leaves the first node that is not "less" than the key. A `rank` array then maps that node back to its sorted index, so the result matches `np.searchsorted`.
 
## API
 
Two layouts, each with a fast variant (extra `rank` array) and a low-memory variant (no `rank` array).
 
### 0-based layout: `n` entries, permuted **in place**
 
Node `k` is stored at `a[k-1]`.
 
```python
import numpy as np
from eytzinger_searchsorted import (
    eytzinger_, eytzinger_searchsorted, eytzinger_searchsorted_norank,
)
 
a = np.sort(np.random.randint(0, 10**9, 1_000_000).astype(np.int32))
sorted_a = a.copy()                       # keep a sorted copy if you still need one
 
rank = eytzinger_(a)                      # a is now in Eytzinger order; returns rank
idx = eytzinger_searchsorted(a, rank, keys)          # == np.searchsorted(sorted_a, keys)
idx = eytzinger_searchsorted_norank(a, keys)         # same result, no rank array
```
 
### 1-based layout: `n + 1` entries, slot 0 left empty
 
Node `k` is stored at `t[k]`, so there is no `-1` offset anywhere. `a` is left untouched and a new array is returned.
 
```python
from eytzinger_searchsorted import (
    eytzinger0, eytzinger0_searchsorted, eytzinger0_searchsorted_norank,
)
 
t, rank = eytzinger0(a_sorted)
idx = eytzinger0_searchsorted(t, rank, keys)
idx = eytzinger0_searchsorted_norank(t, keys)
```
 
All search functions accept `side="left"` (default) or `side="right"`, like `np.searchsorted`.
 
### Rank vs no-rank
 
| Variant | Extra memory | Work per level | Notes |
|---|---|---|---|
| `*_searchsorted` (rank) | `rank` array, `n+1` × 8 bytes | ~4 NumPy passes | One `rank[k]` gather at the end. Faster. |
| `*_searchsorted_norank` | none | ~13 NumPy passes | Computes left-subtree sizes in closed form at every level. |
 
### Build cost
 
Building is a one-time O(n) vectorized scatter, but it needs temporary buffers (about `2n` `intp` entries for the 1-based builder). Building is **not** included in any benchmark number below, so these structures only pay off if you search the same array many times.
 
## Benchmark
 
The benchmark in the script uses random uniform `int32` data and **100,000 random keys**, takes the best of 10 runs, and does **not** time the build. It checks every result against `np.searchsorted` before timing.
 
Results on **NumPy 2.5.0** (ms, lower is better):
 
| n | `np.searchsorted` | 0-based + rank | 0-based, no rank | 1-based + rank | 1-based, no rank |
|---|---|---|---|---|---|
| 4,096 | **0.82** | 2.96 | 5.56 | 2.90 | 5.48 |
| 65,536 | **1.21** | 3.86 | 7.15 | 3.54 | 6.77 |
| 1,048,576 | **2.17** | 4.77 | 8.93 | 4.43 | 8.64 |
| 16,777,216 | 8.58 | 7.56 | 12.29 | **7.12** | 11.91 |
| 268,435,456 | 13.51 | 10.65 | 16.11 | **10.16** | 16.10 |
 
## Ideas to explore
 
- Preallocated buffers and `out=` arguments to remove per-level temporaries.
- A C++ port. Python implementation is already showing promissing results.
- Combining multi-threading with batching within each thread.
