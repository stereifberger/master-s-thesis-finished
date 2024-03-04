from keras.models import Sequential, Model
from keras.layers import Input,Dense, SimpleRNN, TimeDistributed, LSTM, RepeatVector
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random as rd
from random import seed, randint, choice
import numpy as np
from numpy import argmax
import pandas as pd

class Premises:
  def __init__(self,
               term_num = int,
               sample_size = int,
               span_arg_num = list):
    self.term_num = term_num
    self.terms = list(range(1, term_num + 1))
    self.sample_size = sample_size
    self.span_arg_num = span_arg_num
    self.premises = self.input_gen()

  def wff_gen(self):
    arg_num = rd.randint(*self.span_arg_num)
    arguments = rd.sample(self.terms, arg_num)
    # Optionally some logic for building well formed formulas.
    return arguments

  def input_gen(self):
    premises = []
    while len(premises) < self.sample_size:
        random_premises = self.wff_gen()
        if random_premises not in premises:
          premises.append(random_premises)
    return premises
    
class Derivation:
  def __init__(self,
               term_num = int,
               premises = list,
               nonterms = list,
               span_iterations = list):
    self.term_num = term_num
    self.premises = premises
    self.nonterms = nonterms
    self.span_iterations = span_iterations
    self.recursion_task = self.recursion_task()

  def X_gen(self, derivation, premise):
      conclusion = derivation[-1]
      X = flatten([premise, self.term_num + 1, derivation[-1]])
      return X

  def recursion_task(self):
    y = []
    X = []
    for derivation in self.premises:
      derivation_new = []
      iterations = range(rd.randint(*self.span_iterations))
      for iteration in iterations:
        filtered_nonterms = []
        for nonterm in nonterms:
          arg_num = nonterm.__code__.co_argcount # Line by GPT.
          if arg_num <= len(derivation):
            filtered_nonterms.append(nonterm)
        nonterm = rd.choice(filtered_nonterms)
        arg_num = nonterm.__code__.co_argcount
        args = rd.sample(flatten(derivation), arg_num)
        derived = nonterm(*args)
        derivation_new.append(derived)
        if iteration == 0:
          premise = derivation
      y.append(flatten(derivation_new))
      X.append(self.X_gen(derivation_new, premise))
    return X, y
    
class Network_Training:
  def __init__(self,
               self.hidden = int
               self.hidden_dimensions = int
               self.activation = str
               ):
    self.network = self.network()
    self.hidden = hidden
    self.hidden_dimension = hidden_dimension

  """A function with witch architecture and parameters are selected."""
  def network(architecture = str, features_number = int, dim1 = int, dim2 = int):
    """Neural network from towardsdatascience"""
    model = Sequential()
    if architecture == "forward":
      model.add(Dense(16, input_dim=features_number, activation=self.activation))
      for i in range(self.hidden_layers):
            model.add(Dense(self.hidden_dimension, activation=self.activation)) #GPT
      model.add(Dense(features_number, activation=’softmax’))
    if architecture == "rnn":
      model.add(SimpleRNN(128, input_shape=(dim1,dim2), return_sequences = True))
      model.add(SimpleRNN(256, return_sequences = True))
      model.add(SimpleRNN(256, return_sequences = True))
      model.add(SimpleRNN(256, return_sequences = True))
      model.add(SimpleRNN(128, return_sequences = False))
      model.add(Dense(dim1*dim2, activation='sigmoid'))
      model.build()
    if architecture == "lstm":
      model.add(LSTM(units=50, input_shape=(dim1,dim2)))
      model.add(Dense(dim1*dim2))
    return model

def train(model_name = str,
          term_num = int,
          nonterms = list,
          iter_range = list,
          sample_size = int,
          max_length = int):
  num_classes = term_num + 2
  X_y = Derivation(term_num = 4, premises = premises_data, nonterms = nonterms, span_iterations = [1,9])
  X, y = X_y.recursion_task
  X, y = pad(X,y)
  X = to_categorical(X, num_classes)
  y = to_categorical(y, num_classes)
  if model_name == "forward":
    X = out(X)
  y = out(y)
  X = np.array(X)
  y = np.array(y)
  global X_test
  global y_test
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
  features_number = X.shape[1]
  if model_name == "rnn" or model_name == "lstm":
    dim1 = X.shape[1]
    dim2 = X.shape[2]
  else:
    dim1 = 0
    dim2 = 0
  global model
  model = change_model(model_name, features_number, dim1, dim2)
  model.compile(loss="mse", optimizer='adam', metrics=['mae'])
  global history
  history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=200, batch_size=64)
  
def train(model_name = str,
          term_num = int,
          nonterms = list,
          iter_range = list,
          sample_size = int,
          max_length = int):
  num_classes = term_num + 2
  X_y = Derivation(term_num = 4, premises = premises_data, nonterms = nonterms, span_iterations = [1,9])
  X, y = X_y.recursion_task
  X, y = pad(X,y)
  X = to_categorical(X, num_classes)
  y = to_categorical(y, num_classes)
  if model_name == "forward":
    X = out(X)
  y = out(y)
  X = np.array(X)
  y = np.array(y)
  global X_test
  global y_test
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
  features_number = X.shape[1]
  if model_name == "rnn" or model_name == "lstm":
    dim1 = X.shape[1]
    dim2 = X.shape[2]
  else:
    dim1 = 0
    dim2 = 0
  global model
  model = change_model(model_name, features_number, dim1, dim2)
  model.compile(loss="mse", optimizer='adam', metrics=['mae'])
  global history
  history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=200, batch_size=64)