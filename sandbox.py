import timeit
from collections import deque, OrderedDict
from bisect import insort_left
import numpy as np

ceil = 1000000
val = 250000
a = list(np.arange(ceil))

na = np.arange(ceil)

d = deque(np.arange(ceil))

s = timeit.default_timer()
a.pop()
print("remove from regular list: %s" % ((timeit.default_timer() - s) * 100))

# s = timeit.default_timer()
# np.delete(na, val)
# print("remove from numpy array list: %s" % (timeit.default_timer() - s))

s = timeit.default_timer()
d.pop()
print("remove from deque list: %s" % ((timeit.default_timer() - s) * 100))
