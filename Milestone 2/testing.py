import torch
import torch.nn as nn
import numpy as np
import pybullet as p
import pybullet_data
from collision_utils import get_collision_fn

UR5_JOINT_INDICES = [0, 1, 2]
def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)
# Initialize
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(enableFileCaching=0)
p.setGravity(0, 0, -9.8)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)




#This must match the structure in training
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.PReLU(),
            nn.Dropout(0),  # Dropout layer after the first activation
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(0),  # Dropout layer after the second activation
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

#Get Model
model = Net()
model.load_state_dict(torch.load("C:/Users/cqlar/Documents/GitHub/CS558-Project/Milestone 2/models/collision_checker_93.pt"))
print("Model Loaded")

#Load Data
data_directory = "C:/Users/cqlar/Documents/GitHub/CS558-Project/Milestone 1/new_data"

#Give joint angles (q) and obstacles (o) here
q = [-0.37164053,  4.3767815,  -1.5157155]
o = [0.37983316, -0.2885536 ,  0.4557185]


obstacle = p.loadURDF('assets/block.urdf', basePosition=o, useFixedBase=True)
obstacles = [obstacle]
# Initialize Collision Checker
collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                    attachments=[], self_collisions=True,
                                    disabled_collisions=set())
set_joint_positions(ur5, UR5_JOINT_INDICES, q)

#Tell what the correct answer is
correct = collision_fn(q) #0 = no collision; 1 = collision
p.disconnect()
#Calculate Result
input = torch.tensor([q[0], q[1], q[2], o[0], o[1], o[2]])
# output = model(input)
# result = torch.max(output.data, 0)[1].tolist() 

# #Display Result
if correct == 1:
    print("Correct result is Collision")
else:
    print("Correct result is No Collision")

# Looping N times
N = 1
count = 0
for i in range(N):
    output = model(input)
    result = torch.max(output.data, 0)[1].tolist()
    if result == correct:
        count += 1
print(output)
print(f"Accuracy = {100 * count/N:.02f}")

print("\nTESTING\n")
N = 20
v = torch.zeros([(N+1)*2, 6])
v[N,:3] = torch.tensor([1,1,1])
v[(N+1)*2-1, :3] = torch.tensor([1,1,1])
# print(v)
for i in range(0, N):
    v[i,:3] = torch.tensor([0,0,0]) + (0.1 * i)
    v[N+1+i, :3] = torch.tensor([0,0,0]) + (0.1 * i)
v[:N+1,3:] = torch.tensor([.2,.2,.2])
v[N+1:,3:] = torch.tensor([.4, .4, .4])
# print(v)
output= model(v)
# print(output)
# print(output[:,1])
# print(torch.abs(output[:,1]) < 4)
# print(len(torch.abs(output[:,1]) < 4))
result = torch.max(output.data, 1)[1]
print(result)
collisions = (result == 1).tolist()
print(collisions)

if True in (collisions):
    for i in range(0, len(output)):
        if collisions[i] == True:
            result[i] = 4
print(result)



if 1 in result:
    print(True)
else:
    print(False)
print(result)