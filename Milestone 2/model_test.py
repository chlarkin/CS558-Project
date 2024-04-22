# #NOTE Collision = 1 = True.      Collision Free = 0 = False

# import torch
# import torch.nn as nn
# import numpy as np
# import os

# #This must match the structure in training
# class Net(nn.Module): 
#     def __init__(self):
#         super(Net, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(6, 256),
#             nn.PReLU(),
#             nn.Dropout(0.1),  # Dropout layer after the first activation
#             nn.Linear(256, 128),
#             nn.PReLU(),
#             nn.Dropout(0.1),  # Dropout layer after the second activation
#             nn.Linear(128, 64),
#             nn.PReLU(),
#             nn.Linear(64, 2)
#         )

#     def forward(self, x):
#         return self.net(x)
    
#     def reset_weights(self):
#         for layer in self.net:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
#                 if layer.bias is not None:
#                     layer.bias.data.fill_(0.01)

# def load_test_data(directory, test_filenames, mean, std):
#     test_data = []
#     test_labels = []
    
#     for filename in os.listdir(directory):
#         path = os.path.join(directory, filename)
#         if os.path.isfile(path):
#             with open(path, 'r') as file:
                
#                 # Check if current file is designated for testing
#                 if filename in test_filenames:
#                     file_data = []
#                     file_labels = []
#                     for line in file:
#                         parts = line.strip().split(', ')
#                         if len(parts) == 7:
#                             combined_input = [float(part) for part in parts[:-1]]
#                             label = 1 if parts[6] == 'True' else 0
#                             file_data.append(combined_input)
#                             file_labels.append(label)
#                     test_data.extend(file_data)
#                     test_labels.extend(file_labels)
    
#     test_data = np.array(test_data, dtype=np.float32)
#     test_data = (test_data - mean) / std  # Normalize using loaded mean and std

#     test_labels = np.array(test_labels, dtype=np.int64)
    
#     return test_data, test_labels

# #Get Model
# model = Net()
# model.load_state_dict(torch.load("../Milestone 2/models/collision_checker", map_location='cpu'))
# model.eval()

# #Load Data
# data_directory = "../Milestone 1/new_data"
# test_filenames = ["environment_19.txt", "environment_20.txt"]

# # Load mean and std from files
# mean = np.load('mean.npy')
# std = np.load('std.npy')
# test_data, test_labels = load_test_data(data_directory, test_filenames, mean, std)

# correct = 0
# total = 0
# correct_collision = 0
# total_collision = 0
# correct_collision_free = 0
# total_collision_free = 0

# incorrect_inputs = []
# incorrect_labels = []
# correct_labels = []

# for test_input, test_output in zip(test_data, test_labels):
#     total += 1
#     if test_output == 1:
#         total_collision += 1
#     else:
#         total_collision_free += 1
#     output = model(torch.tensor(test_input))
#     result = torch.max(output.data, 0)[1].tolist()

#     if result == test_output:
#         correct += 1
#         if test_output == 1:
#             correct_collision += 1
#         else:
#             correct_collision_free += 1

#     else:
#         incorrect_inputs.append(test_input)
#         incorrect_labels.append(result)
#         correct_labels.append(test_output)

# # print(correct_collision, total_collision, correct_collision_free, total_collision_free, correct, total)
# print(f"\nAccuracy (When Collision occurs): {100*correct_collision/total_collision:.2f}%")
# print(f"Accuracy (When no Collision occurs): {100*correct_collision_free/total_collision_free:.2f}%")
# print(f"Overall Accuracy: {100*correct/total:.2f}%")   

# print("\nHere are the first 10 cases of incorrect results:")
# for i in range(0,10):
#     print(f'Joint Angles:   {incorrect_inputs[i]}       Incorrectly Said:   {incorrect_labels[i]}')


import torch
import torch.nn as nn
import numpy as np
import os

# Define the neural network structure, must match the training structure
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

    def forward(self, x):
        return self.net(x)
    
    def reset_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

def load_test_data(directory, test_filenames, mean, std):
    test_data = []
    test_labels = []
    
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and filename in test_filenames:
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
                test_data.extend(file_data)
                test_labels.extend(file_labels)
    
    test_data = np.array(test_data, dtype=np.float32)
    test_data = (test_data - mean) / std  # Normalize using loaded mean and std

    test_labels = np.array(test_labels, dtype=np.int64)
    
    return test_data, test_labels

# Load the trained model
model = Net()
model.load_state_dict(torch.load("../Milestone 2/models/collision_checker", map_location='cpu'))
model.eval()  # Set the model to evaluation mode

# Load normalization parameters
mean = np.load('mean.npy')
std = np.load('std.npy')

# Load test data
data_directory = "../Milestone 1/new_data"
test_filenames = ["environment_19.txt", "environment_20.txt"]
test_data, test_labels = load_test_data(data_directory, test_filenames, mean, std)

# Evaluate the model on test data
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
    
    output = model(torch.tensor([test_input]))  # Ensure input tensor is properly shaped
    result = torch.max(output.data, 1)[1].item()  # Get the prediction from the network

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

print(f"\nAccuracy (When Collision occurs): {100 * correct_collision / total_collision:.2f}%")
print(f"Accuracy (When no Collision occurs): {100 * correct_collision_free / total_collision_free:.2f}%")
print(f"Overall Accuracy: {100 * correct / total:.2f}%")

print("\nHere are the first 10 cases of incorrect results:")
for i in range(min(10, len(incorrect_inputs))):
    print(f'Joint Angles: {incorrect_inputs[i]} Incorrectly Said: {incorrect_labels[i]}')
