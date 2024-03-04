from tensorflow.keras.utils import to_categorical
from random import seed, randint, choice
import numpy as np
from numpy import argmax
import pandas as pd

def flatten(matrix = list):
  return(list(pd.core.common.flatten(matrix)))
  
"""Pad sublists of lists to same length."""
def pad(list_1 = list, list_2 = list):
  for sublist in list_1:
    while len(sublist) < len(max(list_2, key = len)):
      sublist.append(0)
  for sublist in list_2:
    while len(sublist) < len(max(list_2, key = len)):
      sublist.append(0)
  return list_1, list_2
  
# https://favtutor.com/blogs/partition-list-python
# generator
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i+size]
        
def out(y):
  new = []
  for x in y:
    x = flatten(x)
    new.append(x)
  return new