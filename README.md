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
- **Other memory layouts**, such as the Eytzinger layout, B-tree / S-tree-style blocked layouts.

## Experimentation

- Scripts used for writing [Making np.searchsorted up to 25× Faster in NumPy 2.5](https://blog.scientific-python.org/numpy/searchsorted/) can be found in `experimentation/blogpost` folder.
- Eytzinger layout experiments can be found in `experimentation/eytzinger_numpy` folder

