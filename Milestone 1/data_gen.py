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
# plane = p.loadURDF("plane.urdf")
ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)

def create_environment():
    
    # Generate random positions for obstacles within a predefined range
    obstacle_pos = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.75, 0.75)]
    obstacle = p.loadURDF('assets/block.urdf', basePosition=obstacle_pos, useFixedBase=True)
    
    return obstacle, obstacle_pos

num_env = 250
N = 10000 #Number of Data Points to save

# Loop through environments
for env_index in range(num_env+1):
    print(env_index)
    obstacle, obstacle_pos = create_environment()
    obstacles = [obstacle]
    environment_name = f"environment_{env_index}"
    data_path = f"Milestone 1/data/data/{environment_name}.txt"
    
    # Initialize Collision Checker
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                     attachments=[], self_collisions=True,
                                     disabled_collisions=set())

    q_list = []
    obs_list = []
    c_list = []
    
    for i in range(N):
        # Generate random configuration
        q = sample_conf()
        q_list.append(q)
        obs_list.append(obstacle_pos)

        # Set the robot to the start configuration
        set_joint_positions(ur5, UR5_JOINT_INDICES, q)

        # Check for collision
        c_list.append(collision_fn(q)) #True if collision

    # Save data to file
    with open(data_path, 'w') as file:
        for q, obs, c in zip(q_list,obs_list, c_list):
            file.write(f"{q[0]}, {q[1]}, {q[2]}, {obs[0]}, {obs[1]}, {obs[2]}, {c}\n")

# Close the simulation
p.disconnect()

