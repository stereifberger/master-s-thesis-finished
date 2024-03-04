from keras.models import Sequential, Model
from keras.layers import Input, Dense, SimpleRNN, TimeDistributed, LSTM, RepeatVector
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random as rd
from random import seed, randint, choice, shuffle
import numpy as np
from numpy import argmax
import pandas as pd
from tqdm import tqdm
import copy

"""Define the nonterminals as lambda functions of terms and list them.
The production rules are still very simple:
  - r1: Returns a terminal.
  - r2: Concatenates two terminals.
  - r3: Syntactically like double negation, where a special terminal is iterated
    two times before a terminal and this expression is encapsulated in two
    instances of a special bracket term.
"""
def nonterm_gen(term_num):
  r1 = lambda term: term
  r2 = lambda term1, term2: f"{term_num + 2}{term1}{term2}{term_num + 2}"
  r3 = lambda term1: f"{term_num + 2}{term_num + 3}{term_num + 3}{term1}{term_num + 2}"
  nonterms = [r1, r2, r3]
  global max_num
  max_num = term_num + 10
  return r1, r2, r3, nonterms, max_num

"""A class for generating premises and derivations to conclusions.
Variables:
  - term_num: The number of terminal symbols
  - premises: The premises of a given derivation.
  - iter_range: Number span from which the number of iterations
    with which nonterminals are applied is randomly choosen.
  - sample_size: Number of dervations generated.
  - span_arg_num: Number span from which the number of generated
    premises is randomly chosen."""

class Derivation:
  def __init__(self,
               term_num = int,
               premises = list,
               iter_range = list,
               sample_size = int,
               span_arg_num = list):
    self.term_num = term_num
    self.terms = list(range(1, term_num + 1))
    self.iter_range = iter_range
    self.sample_size = sample_size
    self.span_arg_num =  span_arg_num
    self.premises = self.premises_gen()
    self.recursion_task = self.recursion_task()

  """Generates the premises."""
  def premises_gen(self):
    arg_num = rd.randint(*self.span_arg_num)
    arguments = rd.sample(self.terms, arg_num)
    # Optionally some logic for building well formed formulas.
    return arguments

  """When used with the Derivation method it returns:
    - X: The input to a network of the form [PREMISES, DELIMITER, CONCLUSION]
    - y: The output to a network of the form
      [PREMISES, DERIVATION STEP 1,..., DERIVATION STEP N, CONCLUSION]"""
  def recursion_task(self):
    r1, r2, r3, nonterms, max_num = nonterm_gen(self.term_num)
    X, y = [], []
    while len(X) < self.sample_size:
      derivation_new = []
      derivation = self.premises_gen()
      iterations = range(rd.randint(*self.iter_range))
      premise = copy.copy(derivation)
      for iteration in iterations:
        filtered_nonterms = []
        for nonterm in nonterms:
          arg_num = nonterm.__code__.co_argcount # (Line by GPT)
          if arg_num <= len(derivation):
            filtered_nonterms.append(nonterm)
        nonterm = rd.choice(filtered_nonterms)
        arg_num = nonterm.__code__.co_argcount
        args = rd.sample(flatten(derivation), arg_num)
        derived = nonterm(*args)
        derivation.append(derived)
      conclusion = derivation[-1]
      X_new = to_int(flatten([premise, self.term_num + 1, conclusion]))
      y.append(to_int(flatten(derivation)))
      X.append(X_new)

    return X, y

"""Filters the training and testing examples to the shortest
derivations from a given set of premises and conclusions."""
def filter(X, y): # (Begin GPT, modified my the author)
  to_delete = []
  for i in range(len(X)):
      potential = []
      for t in range(len(X)):
          if tuple(X[t]) == tuple(X[i]):
              potential.append(t)

      if len(potential) > 1:
          shortest_index = min(potential, key=lambda x: len(y[x]))
          for t in sorted(potential, reverse=True):
              if t != shortest_index and t not in to_delete:
                to_delete.append(t)

  X_new, y_new = [], []
  for i in range(len(X)):
    if i not in to_delete:
      X_new.append(X[i])
      y_new.append(y[t])

  return X_new, y_new # (End GPT, modified my the author)
  
architectures = ['forward', 'rnn', 'lstm']

"""A function with witch architecture and parameters are selected.
Variables:
  - architecture: The name of the chosen architecture, for example "lstm".
  - num_hidden: Number of hidden layers.
  - num_hidden_dim: Number of neurons in each hidden layer.
  - func: The activation function used for neurons.
  - dim1, dim2: The first and second dimension of the test and training data.
    First dimension: How many symbols the training/test examples are long each.
    Second dimension: The length of the one-hot-encoding.
"""
def change_model(architecture = str,
                 num_hidden = int,
                 num_hidden_dim = int,
                 func = str,
                 dim1 = int,
                 dim2 = int):

  model = Sequential()
  if architecture == "forward":
    model.add(Dense(60,
                    input_dim = dim1*dim2,
                    activation = func))
    for i in range(num_hidden): # (Loop by GPT)
      model.add(Dense(num_hidden_dim,
                      activation = func))
    model.add(Dense(dim1*dim2,
                    activation=func))

  if architecture == "rnn":
    model.add(SimpleRNN(60,
                        input_shape = (dim1,dim2),
                        return_sequences = True))
    for i in range(num_hidden): # (Loop by GPT)
      model.add(SimpleRNN(num_hidden_dim - 1,
                          activation = func,
                          return_sequences = True))
    model.add(SimpleRNN(num_hidden_dim,
                        return_sequences = False))
    model.add(Dense(dim1*dim2,
                    activation = 'sigmoid'))
    model.build()

  if architecture == "lstm":
    model.add(LSTM(60,
                   input_shape = (dim1,dim2)))
    model.add(Dense(dim1*dim2))
  return model
  
"""Uses "get_training_data()" to generates input and output data,
which is padded to the same length with zeros, one-hot encoded
and spilt into raining and test datasets."""

def get_training_data(term_num = int,
          span_arg_num = list,
          iter_range = list,
          sample_size = int,
          ):

  X, y = Derivation(term_num=term_num,
                    iter_range=iter_range,
                    sample_size=sample_size,
                    span_arg_num=span_arg_num).recursion_task
  X, y = filter(X, y)
  print(f"Sample size: {len(X)}")
  X, y = pad(X, y)
  X = to_categorical(X, max_num)
  y = to_categorical(y, max_num)
  y = out(y)
  X = np.array(X)
  y = np.array(y)
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
  return X_train,X_test,y_train,y_test

"""Takes the training data with all variables, relevant for training,
compiles a model with the architecture specified in the architecture variable,
and returns the history of the training."""

def evaluate(X_train = list,
            X_test = list,
            y_train = list,
            y_test = list,
            architecture = str,
            epochs = int,
            loss = str,
            metrics = list,
            num_hidden = int,
            num_hidden_dim = int,
            func = str):
  dim1 = X_train.shape[1]
  dim2 = X_train.shape[2]
  if architecture == "forward":
    X_train = np.array(out(X_train))
    X_test = np.array(out(X_test))
  model = change_model(architecture, num_hidden, num_hidden_dim, func, dim1, dim2)
  model.compile(loss=loss, optimizer='adam', metrics=metrics)
  history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=epochs)
  return history