import torch
import torch.nn as nn
import numpy



#Load data from milestone 1
data_path = ".../data/..."

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(path):
    #TODO write funciton to load data into variable
    return

#Define class for NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        print('using net')
        self.Net = nn.Sequential(nn.Linear(4, 64),nn.PReLU(),
							   		nn.Linear(64, 32),nn.PReLU(),
							   		nn.Linear(32, 16),nn.PReLU(),
							   		nn.Linear(32, 2), nn.Softmax(dim=0))
    def forward(self, x):
        return self.Net(x) 
    #TODO: determine the output of net (probability of 0/1?)


model = Net()
model = model.to(device)

data = load_data(data_path)
data = torch.tensor(data).to(device)

#TODO: Create loops for training (epochs? batch size? set data aside for testing?)

