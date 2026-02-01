import sys
import timeit
from typing import List, Set

import numpy as np

from src.utils.bitset import BitSet


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def benchmark_storage(elements: List[int], n_universe: int):
    print(f"\n--- Storage Benchmark (n_elements={len(elements)}, universe_size={n_universe}) ---")

    # 1. Python Set
    py_set = set(elements)
    size_set = sys.getsizeof(py_set)  # This is shallow, but for ints it's mostly the structure
    # For more accuracy recursive:
    # size_set = get_size(py_set)

    # 2. BitSet
    bit_set = BitSet(elements)
    size_bitset = sys.getsizeof(bit_set) + sys.getsizeof(bit_set._val)

    # 3. Numpy bool array
    # We need an array of size equal to max element + 1 (or universe size)
    np_arr = np.zeros(n_universe, dtype=bool)
    if elements:
        np_arr[elements] = True
    size_np = np_arr.nbytes

    print(f"{'Type':<15} | {'Size (bytes)':<15} | {'Ratio (vs BitSet)':<15}")
    print("-" * 50)
    print(f"{'BitSet':<15} | {size_bitset:<15} | {1.0:<15.2f}")
    print(f"{'set':<15} | {size_set:<15} | {size_set / size_bitset:<15.2f}")
    print(f"{'numpy (bool)':<15} | {size_np:<15} | {size_np / size_bitset:<15.2f}")


def benchmark_time(n_elements: int, n_universe: int, n_iterations: int = 1000):
    print(
        f"\n--- Time Benchmark (n_elements={n_elements}, universe_size={n_universe}, iters={n_iterations}) ---"
    )

    # Generate random data
    rng = np.random.default_rng(42)
    data1 = rng.choice(n_universe, size=n_elements, replace=False).tolist()
    data2 = rng.choice(n_universe, size=n_elements, replace=False).tolist()

    # Creation
    t_create_set = timeit.timeit(lambda: set(data1), number=n_iterations)
    t_create_bs = timeit.timeit(lambda: BitSet(data1), number=n_iterations)

    # Setup for operations
    s1 = set(data1)
    s2 = set(data2)
    bs1 = BitSet(data1)
    bs2 = BitSet(data2)

    # Union
    t_union_set = timeit.timeit(lambda: s1.union(s2), number=n_iterations)
    t_union_bs = timeit.timeit(lambda: bs1.union(bs2), number=n_iterations)

    # Intersection
    t_inter_set = timeit.timeit(lambda: s1.intersection(s2), number=n_iterations)
    t_inter_bs = timeit.timeit(lambda: bs1.intersection(bs2), number=n_iterations)

    # Membership (check 100 random items)
    check_items = rng.choice(n_universe, size=100, replace=True).tolist()

    def check_set_many():
        for x in check_items:
            x in s1

    def check_bs_many():
        for x in check_items:
            x in bs1

    t_in_set = timeit.timeit(check_set_many, number=n_iterations)
    t_in_bs = timeit.timeit(check_bs_many, number=n_iterations)

    print(f"{'Operation':<20} | {'BitSet (s)':<15} | {'Set (s)':<15} | {'Speedup (Set/BS)':<15}")
    print("-" * 70)
    print(
        f"{'Creation':<20} | {t_create_bs:<15.5f} | {t_create_set:<15.5f} | {t_create_set / t_create_bs:<15.2f}"
    )
    print(
        f"{'Union':<20} | {t_union_bs:<15.5f} | {t_union_set:<15.5f} | {t_union_set / t_union_bs:<15.2f}"
    )
    print(
        f"{'Intersection':<20} | {t_inter_bs:<15.5f} | {t_inter_set:<15.5f} | {t_inter_set / t_inter_bs:<15.2f}"
    )
    print(
        f"{'Membership (100x)':<20} | {t_in_bs:<15.5f} | {t_in_set:<15.5f} | {t_in_set / t_in_bs:<15.2f}"
    )


if __name__ == "__main__":
    # Case 1: Sparse / Small
    benchmark_storage(list(range(0, 100, 2)), 100)  # 50 elements, max 100

    # Case 2: Dense / Medium
    benchmark_storage(list(range(1000)), 1000)  # 1000 elements, max 1000

    # Case 3: Sparse / Large Universe
    # 1000 elements spread over 1,000,000 range
    rng = np.random.default_rng(42)
    sparse_data = rng.choice(1000000, size=1000, replace=False).tolist()
    benchmark_storage(sparse_data, 1000000)

    # Time benchmarks
    print("\n" + "=" * 80)
    benchmark_time(n_elements=100, n_universe=1000, n_iterations=10000)
    benchmark_time(n_elements=1000, n_universe=10000, n_iterations=1000)
    benchmark_time(n_elements=10000, n_universe=100000, n_iterations=100)
