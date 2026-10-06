"""
Eytzinger-layout searchsorted in plain NumPy (no classes, just arrays).
 
  0-based layout (n entries, in place): node k has children 2k, 2k+1, stored at a[k-1]
    rank = eytzinger_(a)                       # a (sorted) is permuted in place; returns rank array
    eytzinger_searchsorted(a, rank, keys)      # == np.searchsorted(original_sorted, keys)
    eytzinger_searchsorted_norank(a, keys)     # same result, no rank array (more work per level)
 
  1-based layout (Algorithmica style, n+1 entries, slot 0 unused): node k stored at t[k]
    t, rank = eytzinger0(a)                    # a is untouched; t has one extra leading slot
    eytzinger0_searchsorted(t, rank, keys)
    eytzinger0_searchsorted_norank(t, keys)
 
rank[k] = sorted index of node k (rank[0] = n, meaning "no element >= key").
"""
import numpy as np
 
 
def _bit_length(x):
    return np.frexp(x)[1].astype(np.intp)       # vectorized int.bit_length
 
 
def eytzinger_(a):
    """Permute sorted 1-D array `a` into Eytzinger order, in place, and return
    `rank` (n+1 intp array: rank[k] = sorted index of node k, rank[0] = n).
    Needs O(n) temporary buffers during the permutation."""
    n = a.shape[0]
    rank = np.full(n + 1, n, dtype=np.intp)
    if n < 2:
        if n == 1:
            rank[1] = 0
        return rank
    L = n.bit_length()
    k = np.arange(1, n + 1, dtype=np.intp)
    d = _bit_length(k) - 1                         # depth of node k
    key = (2 * (k - (1 << d)) + 1) << (L - 1 - d)  # in-order slot in a perfect tree
    slots = np.zeros(1 << L, dtype=np.intp)
    slots[key] = k
    nodes_inorder = slots[slots > 0]               # node ids in sorted order
    a[nodes_inorder - 1] = a.copy()                # sorted[r] -> node nodes_inorder[r]
    rank[nodes_inorder] = np.arange(n, dtype=np.intp)
    return rank
 
 
def eytzinger_searchsorted(a, rank, keys, side="left"):
    """Fast path: 1 gather + 1 compare + shift/add per level, one rank gather at the end."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    cmp = np.less if side == "left" else np.less_equal   # True -> go right
    keys = np.asarray(keys)
    n = a.shape[0]
    if n == 0:
        return np.zeros(keys.shape, dtype=np.intp)
    L = n.bit_length()
    k = np.ones(keys.shape, dtype=np.intp)
 
    for _ in range(L - 1):                         # node k always exists here
        k = 2 * k + cmp(a[k - 1], keys)
 
    valid = k <= n                                 # last level: k may not exist
    k = 2 * k + (~valid | cmp(a[np.minimum(k, n) - 1], keys))
 
    # strip trailing right-moves and the last left-move -> first node not "less"
    t = ~k & (k + 1)                               # isolate lowest 0 bit
    k >>= _bit_length(t)
    return rank[k]
 
 
def eytzinger_searchsorted_norank(a, keys, side="left"):
    """No rank array: the sorted index is accumulated while descending
    (closed-form left-subtree sizes). Less memory, ~3x the work per level."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    cmp = np.less if side == "left" else np.less_equal
    keys = np.asarray(keys)
    n = a.shape[0]
    res = np.zeros(keys.shape, dtype=np.intp)
    if n == 0:
        return res
    L = n.bit_length()
    k = np.ones(keys.shape, dtype=np.intp)
    for i in range(L - 1):
        right = cmp(a[k - 1], keys)
        m = L - 2 - i
        left_size = ((1 << m) - 1) + np.clip(n - ((2 * k) << m) + 1, 0, 1 << m)
        res += right * (left_size + 1)
        k = 2 * k + right
    valid = k <= n
    res += valid & cmp(a[np.minimum(k, n) - 1], keys)
    return res
 
 
# ---------------------------------------------------------------------------
# 1-based layout: slot 0 is left empty, node k lives at t[k]  (no "-1" anywhere)
# ---------------------------------------------------------------------------
def eytzinger0(a):
    """Return (t, rank). t has n+1 entries: t[0] is an unused dummy, t[1:] holds
    `a` in Eytzinger order. rank[k] = sorted index of node k, rank[0] = n."""
    n = a.shape[0]
    t = np.zeros(n + 1, dtype=a.dtype)
    rank = np.full(n + 1, n, dtype=np.intp)
    if n == 0:
        return t, rank
    L = n.bit_length()
    k = np.arange(1, n + 1, dtype=np.intp)
    d = _bit_length(k) - 1
    key = (2 * (k - (1 << d)) + 1) << (L - 1 - d)
    slots = np.zeros(1 << L, dtype=np.intp)
    slots[key] = k
    nodes_inorder = slots[slots > 0]
    t[nodes_inorder] = a
    rank[nodes_inorder] = np.arange(n, dtype=np.intp)
    return t, rank
 
 
def eytzinger0_searchsorted(t, rank, keys, side="left"):
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    cmp = np.less if side == "left" else np.less_equal   # True -> go right
    keys = np.asarray(keys)
    n = t.shape[0] - 1
    if n == 0:
        return np.zeros(keys.shape, dtype=np.intp)
    L = n.bit_length()
    k = np.ones(keys.shape, dtype=np.intp)
 
    for _ in range(L - 1):                         # node k always exists here
        k = 2 * k + cmp(t[k], keys)
 
    k = 2 * k + ((k > n) | cmp(t[np.minimum(k, n)], keys))   # last level: k may be > n
 
    k >>= _bit_length(~k & (k + 1))                # strip trailing 1s and the last 0
    return rank[k]
 
 
def eytzinger0_searchsorted_norank(t, keys, side="left"):
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    cmp = np.less if side == "left" else np.less_equal
    keys = np.asarray(keys)
    n = t.shape[0] - 1
    res = np.zeros(keys.shape, dtype=np.intp)
    if n == 0:
        return res
    L = n.bit_length()
    k = np.ones(keys.shape, dtype=np.intp)
    for i in range(L - 1):
        right = cmp(t[k], keys)
        m = L - 2 - i
        left_size = ((1 << m) - 1) + np.clip(n - ((2 * k) << m) + 1, 0, 1 << m)
        res += right * (left_size + 1)
        k = 2 * k + right
    res += (k <= n) & cmp(t[np.minimum(k, n)], keys)
    return res
 
 
if __name__ == "__main__":
    import time
    print("numpy", np.__version__)
    rng = np.random.default_rng(0)
 
    # ---- correctness ----
    for n in list(range(0, 70)) + [1000, 4095, 4096, 4097, 100_003]:
        s = np.sort(rng.integers(-50, 50, n).astype(np.int32))
        q = rng.integers(-60, 60, 500).astype(np.int32)
        t = s.copy()
        rank = eytzinger_(t)
        for side in ("left", "right"):
            ref = np.searchsorted(s, q, side)
            assert np.array_equal(eytzinger_searchsorted(t, rank, q, side), ref), (n, side)
            assert np.array_equal(eytzinger_searchsorted_norank(t, q, side), ref), (n, side)
            t1, r1 = eytzinger0(s)
            assert np.array_equal(eytzinger0_searchsorted(t1, r1, q, side), ref), (n, side)
            assert np.array_equal(eytzinger0_searchsorted_norank(t1, q, side), ref), (n, side)
    print("correctness OK")
 
    # ---- benchmark (build is NOT timed: done once before the timing loops) ----
    def best(f, reps=10):
        r = []
        for _ in range(reps):
            s = time.perf_counter(); f(); r.append(time.perf_counter() - s)
        return min(r)
 
    hdr = ["np.searchsorted", "inplace+rank", "inplace", "eyt0+rank", "eyt0"]
    print(f"{'n':>12} " + " ".join(f"{h:>15}" for h in hdr) + "   (ms, 100k keys)")
    for p in (12, 16, 20, 24, 28):
        n = 1 << p
        s = np.sort(rng.integers(0, 2**31 - 1, n, dtype=np.int32))
        q = rng.integers(0, 2**31 - 1, 100_000, dtype=np.int32)
        t = s.copy(); rank = eytzinger_(t)        # builds, untimed
        t1, r1 = eytzinger0(s)
        ref = np.searchsorted(s, q)
        assert np.array_equal(eytzinger_searchsorted(t, rank, q), ref)
        assert np.array_equal(eytzinger_searchsorted_norank(t, q), ref)
        assert np.array_equal(eytzinger0_searchsorted(t1, r1, q), ref)
        assert np.array_equal(eytzinger0_searchsorted_norank(t1, q), ref)
        fs = [lambda: np.searchsorted(s, q),
              lambda: eytzinger_searchsorted(t, rank, q),
              lambda: eytzinger_searchsorted_norank(t, q),
              lambda: eytzinger0_searchsorted(t1, r1, q),
              lambda: eytzinger0_searchsorted_norank(t1, q)]
        print(f"{n:>12} " + " ".join(f"{best(f)*1e3:>15.2f}" for f in fs))
