import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.PReLU(),
            nn.Dropout(0),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(0),
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
            else:
                data.extend(file_data)
                labels.extend(file_labels)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(test_data, dtype=np.float32), np.array(test_labels, dtype=np.int64)

def plot_learning_curves(history):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['collision_accuracy'], label='Collision Accuracy')
    plt.title('Collision Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_class_weights(labels):
    # 'balanced' mode makes the function return the inverse of the class frequencies
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_directory = "../Milestone 1/new_data"
model_path = "../Milestone 2/models/collision_checker"
data, labels, test_data, test_labels = load_data(data_directory)
labels_tensor = torch.tensor(labels, dtype=torch.long)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
epochs = 50
unique_classes = np.unique(labels)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
    print(f"Fold {fold+1}")
    X_train, X_val = data[train_idx], data[val_idx]
    y_train, y_val = labels_tensor[train_idx], labels_tensor[val_idx]
    X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train)
    val_dataset = TensorDataset(X_val_tensor, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Net().to(device)
    model.reset_weights()
        
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [], 'collision_accuracy': []}
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
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        collision_correct = 0
        collision_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                # Calculate collision accuracy
                collision_mask = labels == 1
                collision_correct += (predicted[collision_mask] == labels[collision_mask]).sum().item()
                collision_total += collision_mask.sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total if total > 0 else 0
        collision_accuracy = 100 * collision_correct / collision_total if collision_total > 0 else 0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(100 * correct / total)
        history['val_accuracy'].append(accuracy)
        history['collision_accuracy'].append(collision_accuracy)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Collision Accuracy: {collision_accuracy:.2f}%')

    plot_learning_curves(history)
    fold_results.append({'fold': fold+1, 'train_loss': train_loss, 'val_loss': val_loss, 'accuracy': accuracy, 'collision_accuracy': collision_accuracy})
    torch.save(model.state_dict(), f"{model_path}_fold_{fold+1}.pt")

for result in fold_results:
    print(f"Fold {result['fold']}: Training Loss: {result['train_loss']:.4f}, Validation Loss: {result['val_loss']:.4f}, Accuracy: {result['accuracy']:.2f}%, Collision Accuracy: {result['collision_accuracy']:.2f}%")
