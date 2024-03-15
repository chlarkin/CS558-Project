import pybullet as p
import pybullet_data
import numpy as np

UR5_JOINT_INDICES = [0, 1, 2]

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)

def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id

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

# Generate random congifuration

# Generate obstacle locations
obstacle_positions = []
for _ in range(2):
    obstacle_position = np.random.uniform(-1, 1, size=3)
    obstacle = p.loadURDF('assets/block.urdf',
                          basePosition=obstacle_position.tolist(),
                          useFixedBase=True)
    obstacle_positions.append(obstacle_position)

# generate random start and goal    
start_conf = np.random.uniform(-np.pi, np.pi, size=3)
goal_conf = np.random.uniform(-np.pi, np.pi, size=3)

# Generate random start and goal positions
start_position = np.random.uniform(-1, 1, size=3)
goal_position = np.random.uniform(-1, 1, size=3)
goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])

# Set the robot to the start configuration
set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

# Check for collision

# Save data in correct format for training NN (Joint angles, obstacle locations, collision_flag)