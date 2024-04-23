import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import pybullet as p
import pybullet_data
import numpy as np
import time
from collision_utils import get_collision_fn
import math

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(9, 64),
#             nn.PReLU(),
#             nn.Dropout(0.1),  # Dropout layer after the first activation
#             nn.Linear(64, 32),
#             nn.PReLU(),
#             nn.Dropout(0.1),  # Dropout layer after the second activation
#             nn.Linear(32, 16),
#             nn.PReLU(),
#             nn.Linear(16, 2)
#         )

#     def forward(self, x):
#         return self.net(x)
    
#     def reset_weights(self):
#         for layer in self.net:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
#                 if layer.bias is not None:
#                     layer.bias.data.fill_(0.01)

# model = Net()
# model.reset_weights
# model.load_state_dict(torch.load("C:/Users/cqlar/Documents/GitHub/CS558-Project/Milestone 2/models/collision_checker"))

# input = torch.tensor([-1.2849813223548585, 0.2761616514784402, 1.0698085168065283, -0.48606829713512445, -0.4742208703810389, 0.2026271262856553, 0.4891267201053682, 0.07522346831553983, 0.5050351665101648])
# input2 = torch.tensor([4.64523807145437, 5.496068954524816, -0.9241130145442544, -0.48606829713512445, -0.4742208703810389, 0.2026271262856553, 0.4891267201053682, 0.07522346831553983, 0.5050351665101648])

# print(model(input))
# print(model(input2))
correct = 0
labels = torch.tensor([0,1,0,0,0,0,1])
output = torch.tensor([0,0,0,0,0,0,0])

correct += (output == labels).sum().item()
print(correct)