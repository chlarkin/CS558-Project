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

def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id

def sample_conf():
    sample_joint1 = np.random.uniform(-2*math.pi,2*math.pi)
    sample_joint2 = np.random.uniform(-2*math.pi,2*math.pi)
    sample_joint3 = np.random.uniform(-math.pi,math.pi)
        
    rand_conf = (sample_joint1, sample_joint2, sample_joint3)    
    return rand_conf #Random

# Initialize
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(enableFileCaching=0)
p.setGravity(0, 0, -9.8)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))


# Load Objects
plane = p.loadURDF("plane.urdf")
ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)

#Choose Environment
obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
obstacles = [plane, obstacle1, obstacle2]

# Initialize Collision Checker
collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                    attachments=[], self_collisions=True,
                    disabled_collisions=set())

N = 100 #Number of Data Points to save

q_list = []
c_list = []
for i in range(N):
    # Generate random congifuration
    q = sample_conf()
    q_list.append(q)

    # Set the robot to the start configuration
    set_joint_positions(ur5, UR5_JOINT_INDICES, q)
    time.sleep(.1)

    # Check for collision
    c_list.append(collision_fn(q)) #True if collision

    # Save data in correct format for training NN (Joint angles, obstacle locations, collision_flag)
data = zip(q_list, c_list)
zipped_data = list(data)
# for i in range(len(zipped_data)):
#     print(zipped_data[i])


# Need to figure out how to save the (zipped_data) to a file that can be accessed
# Need to create more environments and save the data

