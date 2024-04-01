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
        flat.append(calc.symb["LB"])
    
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
        flat.append(calc.symb["RB"])
    
    return flat


def reve(inpt):
    x_train_3d = inpt.view(len(inpt), int(len(inpt[0])/(t_nu+8)), t_nu+8)
    inpt_reversed = []
    for i in x_train_3d:
        reverse_x_train = torch.argmax(i, dim=1)
        brackets = [t_nu + 2, t_nu +3]
        reverse_x_train = reverse_x_train.tolist()
        reverse_x_train = ["[" if item == (t_nu + 2) else item for item in reverse_x_train]
        reverse_x_train = ["]" if item == (t_nu + 3) else item for item in reverse_x_train]
        reverse_x_train = [str(item) for item in reverse_x_train]
        stri = ", ".join(str(item) for item in reverse_x_train)
        stri = "[" + stri + "]"
        stri = stri.replace("[,", "[").replace(", ]", "]")
        reverse_x_train = eval(stri)
        inpt_reversed.append(stri)
    return inpt_reversed