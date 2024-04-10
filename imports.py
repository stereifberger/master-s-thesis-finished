#Values:
t_nu = 5

from functools import partial
from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader, random_split

# NumPy
import numpy as np
from numpy import argmax

# PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.functional import pairwise_distance

# Other libraries
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from random import seed, randint, choice, shuffle, sample, random
import pandas as pd
import copy
import re
import importlib
import calculi, util, generation, architectures, losses_old
from ipywidgets import IntText
from IPython.display import display
import time
import math
import copy

