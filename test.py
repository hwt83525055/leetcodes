from itertools import tee
from random import shuffle

def in_order(my_list):
    it1, it2 = tee(my_list)
    it2.__next__()
    return all(a <= b for a, b in zip(it1, it2))

def bogo_sort(array):
    while not in_order(array):
        shuffle(array)
    return array

print(bogo_sort([5, 6, 21, 3, 5, 12, 34, 52, 5]))