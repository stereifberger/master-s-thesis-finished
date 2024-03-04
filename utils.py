import pandas as pd

"""Flatten a matrix to one dimension."""
def flatten(matrix = list):
  return(list(pd.core.common.flatten(matrix)))
  
"""Pad sublists of lists to same length. Necessary because in the task the
output is longer than the input which is not possible in some architectures."""
def pad(list_1 = list, list_2 = list):
  for sublist in list_1:
    while len(sublist) < len(max(list_2, key = len)):
      sublist.append(0)
  for sublist in list_2:
    while len(sublist) < len(max(list_2, key = len)):
      sublist.append(0)
  return list_1, list_2

"""Flatten all subsets of a list"""
def out(y):
  new = []
  for x in y:
    x = flatten(x)
    new.append(x)
  return new

"""The second function uses the first to partition lists into subsets of length n.
(https://www.geeksforgeeks.org/python-program-to-get-all-subsets-of-given-size-of-a-set/)"""
def subsets(numbers):
    if numbers == []:
        return [[]]
    x = subsets(numbers[1:])
    return x + [[numbers[0]] + y for y in x]

def subsets_with_length(numbers, n):
    return [x for x in subsets(numbers) if len(x)==n]

"""Numbers in list encoded as str or int are listed as single digit ints."""
def to_int(data = list):
  new_data = []
  for i in data:
    for t in str(i):
      new_data.append(int(t))
  return new_data