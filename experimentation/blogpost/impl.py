import numpy as np


def searchsorted_py(a, xs):
    res = np.empty(len(xs), dtype=np.int32)

    for i, x in enumerate(xs):
        lo = 0
        hi = len(a)

        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid

        res[i] = lo

    return res


def searchsorted_py_np(a, xs):
    lo = np.zeros(xs.shape, dtype=np.int32)
    hi = np.full(xs.shape, len(a), dtype=np.int32)

    while True:
        active = lo < hi
        if not np.any(active):
            break

        mid = (lo + hi) // 2

        # only index where active
        mid_a = mid[active]
        xs_a = xs[active]

        mask = a[mid_a] < xs_a

        lo[active] = np.where(mask, mid_a + 1, lo[active])
        hi[active] = np.where(mask, hi[active], mid_a)

    return lo


def searchsorted_py_np_fixed(a, xs):
    n = len(a)
    lo = np.zeros(xs.shape, dtype=np.int32)
    hi = np.full(xs.shape, n, dtype=np.int32)

    for _ in range(int(np.ceil(np.log2(n)))):
        mid = (lo + hi) // 2
        go_right = xs > a[mid]
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
    return hi


def searchsorted_py_np_fast_where(arr, keys):
    K = keys.shape[0]
    length = arr.shape[0]

    base = np.zeros(K, dtype=np.intp)

    # Invariant: the insertion index lies in [base, base + length]
    while length > 1:
        half = length >> 1
        length -= half
        mid = base + half
        base = np.where(keys > arr[mid], mid, base)

    # Final step: result is either base and base+1
    base = np.where(keys > arr[base], base + 1, base)

    return base


def searchsorted_py_np_fast_where_no_final(arr, keys):
    K = keys.shape[0]
    length = arr.shape[0]

    base = np.zeros(K, dtype=np.intp)

    while length > 1:
        half = length >> 1
        base = np.where(
            arr[base + half - 1] < keys,
            base + half,
            base,
        )
        length -= half

    return base + (arr[base] < keys)


IMPLEMENTATIONS = {
    "searchsorted_py": searchsorted_py,
    "searchsorted_py_np": searchsorted_py_np,
    "searchsorted_py_np_fixed": searchsorted_py_np_fixed,
    "searchsorted_py_np_fast_where": searchsorted_py_np_fast_where,
    "searchsorted_py_np_fast_where_no_final": searchsorted_py_np_fast_where_no_final,
    "searchsorted_np_2.4": np.searchsorted
}
