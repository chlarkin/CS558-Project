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
    rand_conf = (0,0,0) 
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

def create_environment(index):
    obstacle_positions = []
    for _ in range(2):  # Number of obstacles (assuming 2 obstacles per environment)
        # Generate random positions for obstacles within a predefined range
        obstacle_pos = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(0, 0.75)]
        print(obstacle_pos)
        obstacle_positions.append(obstacle_pos)

    obstacles = []
    for obstacle_pos in obstacle_positions:
        obstacle = p.loadURDF('assets/block.urdf', basePosition=obstacle_pos, useFixedBase=True)
        obstacles.append(obstacle)
    
    return obstacles

# def create_complex_environment(client_id):
#     p.setGravity(0, 0, -10, physicsClientId=client_id)
    
#     # Load the plane and robot UR5
#     p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
#     plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
#     robot_id = p.loadURDF("ur5.urdf", basePosition=[0, 0, 0.02], physicsClientId=client_id)
    
#     # Create random obstacles
#     num_obstacles = 20
#     for _ in range(num_obstacles):
#         # Generate random position and size
#         pos = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(0.5, 2)]
#         size = [np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5)]
        
#         # Load obstacle with random orientation
#         orientation = p.getQuaternionFromEuler([np.random.uniform(0, np.pi), np.random.uniform(0, np.pi), np.random.uniform(0, np.pi)])
#         obstacle_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size, physicsClientId=client_id)
#         p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_id, basePosition=pos, baseOrientation=orientation, physicsClientId=client_id)
    
    
#     return plane_id, robot_id
# Loop through environments
# Usage
# client_id = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
# create_complex_environment(client_id)
obstacles = create_environment(1)
print(obstacles)

# Initialize Collision Checker
collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                    attachments=[], self_collisions=True,
                                    disabled_collisions=set())

q_list = []
c_list = []

# Generate random configuration
q = sample_conf()
q_list.append(q)

# Set the robot to the start configuration
set_joint_positions(ur5, UR5_JOINT_INDICES, q)
time.sleep(10)

# Check for collision
c_list.append(collision_fn(q)) #True if collision

# Close the simulation
p.disconnect()