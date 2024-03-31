from libs import *

"""Flatten a matrix to one dimension."""
def flatten(matrix = list):
  return(list(pd.core.common.flatten(matrix)))

"""Pad sublists of lists to same length. Necessary because in the task the
output is longer than the input which is not possible in some architectures."""
def pad(inpt = list):
    for sublist in tqdm(inpt, desc="Padded exmaples from X"):
        while len(sublist) < len(max(inpt, key = len)):
            sublist.append(0)
    return inpt

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
  new_data = [0]
  for i in data:
    for t in str(i):
      new_data.append(int(t))
    new_data.append(0)
  return new_data

# Changed from GPT-4
def lflt(inpt = list, insd=False):
    # This list will hold the flattened version of the input_list
    flat = []
    
    # If we're processing elements inside a sublist, add the opening symbol
    if insd:
        flat.append(calc.symb["DE"])
    
    for item in inpt:
        # If the current itemsymb["DE"] is a list, we need to recursively flatten it
        if isinstance(item, list):
            # Extend the current flattened list with the recursively flattened sublist
            # Notice we pass 'True' for the 'inside' flag to handle sublist brackets
            flat.extend(lflt(item, True))
        else:
            # If it's not a list, just append the item to the flattened list
            flat.append(item)

    # If we're processing elements inside a sublist, add the closing symbol
    if insd:
        flat.append(calc.symb["DE"])
    
    return flat