# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# import os



# #Load data from milestone 1
# datapath = "../Milestone 1/data"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# import numpy as np

# def normalize_data(data):
#     """
#     Normalize the input data to have a mean of 0 and a standard deviation of 1.
    
#     Parameters:
#     - data: numpy array of shape (n_samples, n_features)
    
#     Returns:
#     - normalized_data: numpy array of shape (n_samples, n_features), normalized
#     """
#     # Compute the mean and std of the dataset
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
    
#     # Avoid division by zero
#     std[std == 0] = 1
    
#     # Normalize the data
#     normalized_data = (data - mean) / std
    
#     return normalized_data

# def load_data(directory):
#     data = []
#     labels = []
#     for filename in os.listdir(directory):
#         path = os.path.join(directory, filename)
#         if os.path.isfile(path):
#             with open(path, 'r') as file:
#                 for line in file:
#                     parts = line.strip().split(', ')
#                     if len(parts) == 4:
#                         angles = [float(parts[0]), float(parts[1]), float(parts[2])]
#                         label = 1 if parts[3] == 'True' else 0
#                         data.append(angles)
#                         labels.append(label)
#     data = np.array(data, dtype=np.float32)
#     labels = np.array(labels, dtype=np.int64)
#     return data, labels


# #Define class for NN
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(3, 64),  # Adjust input features to 3 for joint angles
#             nn.PReLU(),
#             nn.Linear(64, 32),
#             nn.PReLU(),
#             nn.Linear(32, 16),
#             nn.PReLU(),
#             nn.Linear(16, 2)  # Removed the Softmax layer
#         )

#     def forward(self, x):
#         return self.net(x)



# model = Net()
# model = model.to(device)

# data, labels = load_data(datapath)
# normalized_data = normalize_data(data)
# data, labels = torch.tensor(normalized_data).to(device), torch.tensor(labels).to(device)

# # Splitting data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# # Converting arrays to datasets
# train_dataset = TensorDataset(X_train, y_train)
# test_dataset = TensorDataset(X_test, y_test)

# # DataLoader setup
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# #TODO: Create loops for training (epochs? batch size? set data aside for testing?)
# # Training loop
# epochs = 100
# for epoch in range(epochs):
#     model.train()
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# # Testing loop for accuracy calculation
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the test data: {100 * correct / total}%')

import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.PReLU(),
            nn.Dropout(0.1),  # Dropout layer after the first activation
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(0.1),  # Dropout layer after the second activation
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)
    
    def reset_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)


def normalize_data(data):
    """
    Normalize the input data to have a mean of 0 and a standard deviation of 1.
    
    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    
    Returns:
    - normalized_data: numpy array of shape (n_samples, n_features), normalized
    """
    # Compute the mean and std of the dataset
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    # Normalize the data
    normalized_data = (data - mean) / std
    
    return normalized_data

def load_data(directory, test_filenames):
    data = []
    labels = []
    test_data = []
    test_labels = []
    
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            with open(path, 'r') as file:
                file_data = []
                file_labels = []
                for line in file:
                    parts = line.strip().split(', ')
                    if len(parts) == 7:
                        # angles = [float(parts[0]), float(parts[1]), float(parts[2])]
                        # obstacle1 = [float(parts[3]), float(parts[4]), float(parts[5])]
                        # obstacle2 = [float(parts[6]), float(parts[7]), float(parts[8])]
                        combined_input = [float(part) for part in parts[:-1]]
                        label = 1 if parts[6] == 'True' else 0
                        file_data.append(combined_input)
                        file_labels.append(label)
                
                # Check if current file is designated for testing
                if filename in test_filenames:
                    test_data.extend(file_data)
                    test_labels.extend(file_labels)
                else:
                    data.extend(file_data)
                    labels.extend(file_labels)
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    test_data = np.array(test_data, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int64)
    
    return data, labels, test_data, test_labels


device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_filenames = ["environment_19.txt", "environment_20.txt"]
data_directory = "C:/Users/cqlar/Documents/GitHub/CS558-Project/Milestone 1/new_data"
model_path = "C:/Users/cqlar/Documents/GitHub/CS558-Project/Milestone 2/models/collision_checker"

data, labels, test_data, test_labels = load_data(data_directory, test_filenames)
data_normalized = normalize_data(data)

# Convert the labels to a tensor
labels_tensor = torch.tensor(labels, dtype=torch.long)

# K-Fold Cross-Validation
#kf = KFold(n_splits=5)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

fold_results = []

epochs = 100
for fold, (train_idx, val_idx) in enumerate(skf.split(data_normalized, labels)):
    print(f"Fold {fold+1}")
    
    # Preparing the data loaders
    X_train, X_val = data_normalized[train_idx], data_normalized[val_idx]
    y_train, y_val = labels_tensor[train_idx], labels_tensor[val_idx]
    X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train)
    val_dataset = TensorDataset(X_val_tensor, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Resetting the model
    model = Net().to(device)
    model.reset_weights()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Average loss for this epoch
        train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss}')

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Average loss and accuracy for this fold
    val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss}, Accuracy: {accuracy}%')

    # Storing the results
    fold_results.append({'fold': fold+1, 'train_loss': train_loss, 'val_loss': val_loss, 'accuracy': accuracy})
    torch.save(model.state_dict(), model_path)

# Displaying the results for all folds
for result in fold_results:
    print(f"Fold {result['fold']}: Training Loss: {result['train_loss']:.4f}, "
          f"Validation Loss: {result['val_loss']:.4f}, Accuracy: {result['accuracy']:.2f}%")


