#Values:
t_nu = 9

from keras.models import Sequential, Model
from keras.layers import Input, Dense, SimpleRNN, TimeDistributed, LSTM, RepeatVector
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random as rd
from random import seed, randint, choice, shuffle, sample
import numpy as np
from numpy import argmax
import pandas as pd
from tqdm import tqdm
import copy
import re
import importlib
import util, calc, gnrn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_diabetes