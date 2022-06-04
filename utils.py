'''
Parallelization
'''

import multiprocessing as mp
from multiprocessing.pool import ThreadPool


def threadify(f, arr):
    # Maps array with function f parallelized
    pool = ThreadPool(mp.cpu_count())
    result = pool.map(f, arr)
    pool.terminate()
    pool.join()
    return result
