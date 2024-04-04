#Values:
t_nu = 5

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
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import calc, util, gnrn, arch
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import load_diabetes