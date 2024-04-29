#NOTE Collision = 1 = True.      Collision Free = 0 = False

import torch
import torch.nn as nn
import numpy as np
import os
import random

#This must match the structure in training
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.PReLU(),
            nn.Dropout(0.1),  # Dropout layer after the first activation
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(0.1),  # Dropout layer after the second activation
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 2)
        )
        self.obs_pos = []

    def forward(self, x):
        return self.net(x)
    
    def reset_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)
    def add_obs_pos(self, obs_pos):
        self.obs_pos.append(obs_pos)

def load_data(directory, num_test_files=50):
    all_filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    test_filenames = random.sample(all_filenames, num_test_files)
    data = []
    labels = []
    test_data = []
    test_labels = []
    for filename in all_filenames:
        path = os.path.join(directory, filename)
        with open(path, 'r') as file:
            file_data = []
            file_labels = []
            for line in file:
                parts = line.strip().split(', ')
                if len(parts) == 7:
                    combined_input = [float(part) for part in parts[:-1]]
                    label = 1 if parts[6] == 'True' else 0
                    file_data.append(combined_input)
                    file_labels.append(label)
            if filename in test_filenames:
                test_data.extend(file_data)
                test_labels.extend(file_labels)
    return np.array(test_data, dtype=np.float32), np.array(test_labels, dtype=np.int64)

#Get Model
model = Net()
model.load_state_dict(torch.load("../Milestone 2/models/collision_checker_fold_1.pt")) #Path to trained model

#Load Data
data_directory = "../Milestone 1/new_data" #Path to data
test_data, test_labels = load_data(data_directory, 500)

correct = 0
total = 0
correct_collision = 0
total_collision = 0
correct_collision_free = 0
total_collision_free = 0

incorrect_inputs = []
incorrect_labels = []
correct_labels = []

for test_input, test_output in zip(test_data, test_labels):
    total += 1
    if test_output == 1:
        total_collision += 1
    else:
        total_collision_free += 1
    output = model(torch.tensor(test_input))
    result = torch.max(output.data, 0)[1].tolist()

    if result == test_output:
        correct += 1
        if test_output == 1:
            correct_collision += 1
        else:
            correct_collision_free += 1

    else:
        incorrect_inputs.append(test_input)
        incorrect_labels.append(result)
        correct_labels.append(test_output)

# print(correct_collision, total_collision, correct_collision_free, total_collision_free, correct, total)
print(f"\nAccuracy (When Collision occurs): {100*correct_collision/total_collision:.2f}%")
print(f"Accuracy (When no Collision occurs): {100*correct_collision_free/total_collision_free:.2f}%")
print(f"Overall Accuracy: {100*correct/total:.2f}%")   

print("\nHere are 10 random cases of incorrect results:")
for i in range(0,10):
    index = random.randint(0, len(incorrect_inputs))
    print(f'Joint Angles:   {incorrect_inputs[index]}       Incorrectly Said:   {incorrect_labels[index]}')
