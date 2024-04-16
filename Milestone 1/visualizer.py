joint_angles = (-6.141339  ,  4.5743155 , -2.486333   ) #Ex. (0,0,0)
obs_pos = (0.04426698,  0.4595536,   0.31150365)
env = 0 #index of env (0 - 20)

import pybullet as p
import pybullet_data
import numpy as np
import time
from collision_utils import get_collision_fn
import math

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


# Load Objects
# plane = p.loadURDF("plane.urdf")
ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)

obstacle = p.loadURDF('assets/block.urdf', basePosition=obs_pos, useFixedBase=True)
obstacles = [obstacle]

# Initialize Collision Checker
collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                    attachments=[], self_collisions=True,
                                    disabled_collisions=set())

# Generate random configuration
q = joint_angles
# print(f"Joint Angles: {q}")

# print(p.getNumJoints(ur5))

# Set the robot to the start configuration
set_joint_positions(ur5, UR5_JOINT_INDICES, q)

# Check for collision
print(f"Collision? {collision_fn(q)}") #True if collision

# Close the simulation
time.sleep(10)
p.disconnect()