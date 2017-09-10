import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import numpy as np



net = torch.load("./data/net_saved/net_82_94.pkl")

print(net)