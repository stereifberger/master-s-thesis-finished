import pandas as pd

"""Flatten a matrix to one dimension."""
def flatten(matrix = list):
  return(list(pd.core.common.flatten(matrix)))

def plus(x, y):
  return x + y