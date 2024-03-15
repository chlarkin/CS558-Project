from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import math
import os
import sys
import random


UR5_JOINT_INDICES = [0, 1, 2]


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def remove_marker(marker_id):
   p.removeBody(marker_id)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    args = parser.parse_args()
    return args

#your implementation starts here
#refer to the handout about what the these functions do and their return type
###############################################################################
class RRT_Node:
    def __init__(self, conf):
        self.conf = conf
        self.children = []
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

# Sample in a way that it lets you reach to the goal node.


# this function should return a robot conf uniformly sampled from the robot’s 
# joint limits along with a boolean flag indicating whether the sampled conf 
# is goal conf
def sample_conf():

    # randomly sample 3 joints
    rand_joint_1 = np.random.uniform(2*-np.pi, 2*np.pi, 1)
    rand_joint_2 = np.random.uniform(2*-np.pi, 2*np.pi, 1)
    rand_joint_3 = np.random.uniform(-np.pi, np.pi, 1)
    
    # Random configuration
    rand_conf = [*rand_joint_1, *rand_joint_2, *rand_joint_3]


    # check if rand conf is goal
    is_goal = np.linalg.norm(np.array(rand_conf) - np.array(goal_conf)) < 0.3

    rand_node = RRT_Node(rand_conf)
    #print(rand_node)
    return rand_node, is_goal

# return the nearest node of rand node from the
# node list, node list should contain all nodes 
# in the current RRT Tree
def find_nearest(rand_node, node_list):
    dist = float('inf')
    for node in node_list:
        curr_dist = np.linalg.norm(np.array(rand_node.conf) - np.array(node.conf))
        if curr_dist < dist:
            dist = curr_dist
            nearest_node = node
    return nearest_node

# check if there is a collision path between the two
# input nodes. Collision checking needs to be performed
# on each interpolated middle node between the two inputs. 
# Use a step size of 0.05 as that is what we used in the
# grading system        
def steer_to(nearest_node, rand_node):
    # interpolate between two nodes with a step size of 0.05
    step_size = 0.05
    
    # interpolate between node configurations
    direction_vector = np.array(rand_node.conf) - np.array(nearest_node.conf)

    #print(direction_vector)
    # calculate number of steps to reach target
    numsteps = int(np.linalg.norm(direction_vector) / step_size)
    
    #print(np.array(increase_by.shape()))
    step = direction_vector / numsteps
    #print("Step:" , step)
    old_conf = np.array(nearest_node.conf)

    for _ in range(numsteps):
        # Generating intermediate configurations
        new_conf = old_conf + step
        #print(new_conf)
        # check for collision in this configuration
        if (collision_fn(new_conf)):
            return False
        # else:
        #     set_joint_positions(ur5, UR5_JOINT_INDICES, nearest_node.conf)
        #     nearest_state = p.getLinkState(ur5, 3)[0]

        #     set_joint_positions(ur5, UR5_JOINT_INDICES, new_conf)
        #     new_state = p.getLinkState(ur5, 3)[0]
        #     p.addUserDebugLine(nearest_state, new_state, [0,0,1], 1.0)

        old_conf = new_conf

    
    return True

def steer_to_get_path(conf1, conf2):
    intermediate_steps = []
    step_size = 0.05
    direction = np.subtract(conf2, conf1)
    numsteps = int(np.linalg.norm(direction) / step_size)
    step = direction / numsteps
    curr = conf1
    for _ in range(numsteps):
        intermediate_steps.append(curr)
        curr = curr + step
    return intermediate_steps

def steer_to_until(rand_node, nearest_node):
     # interpolate between two nodes with a step size of 0.05
    step_size = 0.05
    
    # interpolate between node configurations
    direction_vector = np.array(rand_node.conf) - np.array(nearest_node.conf)

    #print(direction_vector)
    # calculate number of steps to reach target
    numsteps = int(np.linalg.norm(direction_vector) / step_size)
    
    #print(np.array(increase_by.shape()))
    step = direction_vector / numsteps
    #print("Step:" , step)
    old_conf = np.array(nearest_node.conf)

    for _ in range(numsteps):
        # Generating intermediate configurations
        new_conf = old_conf + step
        #print(new_conf)
        # check for collision in this configuration
        if (collision_fn(new_conf)):
            return old_conf
        # else:
        #     set_joint_positions(ur5, UR5_JOINT_INDICES, nearest_node.conf)
        #     nearest_state = p.getLinkState(ur5, 3)[0]

        #     set_joint_positions(ur5, UR5_JOINT_INDICES, new_conf)
        #     new_state = p.getLinkState(ur5, 3)[0]
        #     p.addUserDebugLine(nearest_state, new_state, [0,0,1], 1.0)

        old_conf = new_conf

    
    return old_conf

    
    pass

def distance_func(conf1, conf2):
    return math.sqrt( ((conf2[0] - conf1[0]) ** 2) + ((conf2[1] - conf1[1]) ** 2) + ((conf2[2] - conf1[2]) ** 2))
# the main function that performs the RRT algorithm. 
# This function should return an ordered list of robot 
# confs that moves the robot arm from the start position
# to the goal position.
def RRT():
    ###############################################
    # TODO your code to implement the rrt algorithm
    ###############################################
    
    # initialize our tree with start node
    start_node = RRT_Node(start_conf)
    node_list = []
    node_list.append(start_node)
    
    
    while True:
        # sampling a random configuration 
        # and get the config only
        rand_node, is_goal = sample_conf()
        # if is_goal:
        #     print("Yes 2")
        # find nearest node in the tree to the rand_node
        nearest_node = find_nearest(rand_node, node_list)

        # Extend the tree towards the node if no collisions
        # found
        flag = steer_to(rand_node, nearest_node)
        #print(new_node)
        
        if flag is True:
            # add the node to the list
            node_list.append(rand_node)

            # setting the parent
            rand_node.set_parent(nearest_node)
            
            # adding new node to the children list
            nearest_node.add_child(rand_node)

            # # change the flags
            # goal_node = RRT_Node(goal_conf)
            # flag2 = steer_to(rand_node, goal_node)
            # print("Flag2: ", flag2)
            # # see if goal is reachable from the random node to converge faster
            # if flag2:
            #     goal_node.set_parent(rand_node)
            #     rand_node.add_child(goal_node)
            #     is_goal = True

            # check if the goal is reached
            if is_goal:
                # Reconstructing the path
                path = []
                #goal_node = RRT_Node(goal_conf)
                node = rand_node
                #print("New node: ", node.conf)
                

                while node is not None:
                    path.append(node.conf)
                    node = node.parent

                path.reverse()
                #print("Path:", path)
                #print(path)
                #set_joint_positions(ur5, )
                return path

# Returns a flag if a node from 1 tree
# can be connected to a node from another tree       
def connect(node, node_list):
    nearest_node = find_nearest(node, node_list)
    flag = steer_to(nearest_node, node)
    return flag

def BiRRT():
    #################################################
    # TODO your code to implement the birrt algorithm
    #################################################

    # need two trees
    # 1st tree starts at start node
    tree_a = []
    start_node = RRT_Node(start_conf)
    tree_a.append(start_node)
    #print("Start node: ", start_node, "\n")
    # 2nd tree starts at end node
    tree_b = []
    goal_node = RRT_Node(goal_conf)
    tree_b.append(goal_node)
    #print("Goal node: ", goal_node, "\n")
    working_on_tree_a = True
    while True:
        if working_on_tree_a:
            # generate a random node for tree a
            rand_node_a, is_goal = sample_conf()
            #print("random node: ", rand_node_a, "\n")

            # Find the nearest node in tree a
            nearest_node_a = find_nearest(rand_node_a, tree_a)

            #print("Nearest node A: ", nearest_node_a, "\n")

            # Can these two nodes be connected
            flag_1 = steer_to(nearest_node_a, rand_node_a)

            if flag_1:
                # add the node to the tree
                tree_a.append(rand_node_a)

                # update the parent
                rand_node_a.set_parent(nearest_node_a)

                # add the child
                nearest_node_a.add_child(rand_node_a)

                # node nearest to list b
                nearest_node_in_b = find_nearest(rand_node_a, tree_b)
                #print("Nearest node in B: ", nearest_node_in_b, "\n")

                # try to steer to that nearest node
                flag_3 = steer_to(nearest_node_in_b, rand_node_a)
                #print("flage 3: ", flag_3)
                if flag_3:
                    path_a = []

                    node_a = rand_node_a
                    while node_a is not None:
                        path_a.append(node_a.conf)
                        node_a = node_a.parent

                    path_b = []
                    node_b = nearest_node_in_b
                    while node_b is not None:
                        path_b.append(node_b.conf)
                        node_b = node_b.parent

                    path_a.reverse()
                    path = path_a + path_b
                    #print("path: ", path)
                    return path        
            working_on_tree_a = False

        else:
            # generate a random node for tree a
            rand_node_b, is_goal = sample_conf()
            #print("Random node B: ", rand_node_b, "\n")

            # Find the nearest node in tree a
            nearest_node_b = find_nearest(rand_node_b, tree_b)
            #print("Nearest node B: ", nearest_node_b, "\n")

            # Can these two nodes be connected
            flag_2 = steer_to(nearest_node_b, rand_node_b)

            if flag_2:
                # add the node to the tree
                tree_b.append(rand_node_b)

                # update the parent
                rand_node_b.set_parent(nearest_node_b)

                # add the child
                nearest_node_b.add_child(rand_node_b)

                # node nearest to list a
                nearest_node_in_a = find_nearest(rand_node_b, tree_a)
                
                # try to steer to that nearest node
                flag_4 = steer_to(nearest_node_in_a, rand_node_b)
               
                if flag_4:
                    path_a = []
                    node_a = nearest_node_in_a
                    while node_a is not None:
                        path_a.append(node_a.conf)
                        node_a = node_a.parent

                    path_b = []
                    node_b = rand_node_b
                    while node_b is not None:
                        path_b.append(node_b.conf)
                        node_b = node_b.parent
                    
                    path_a.reverse()
                    path = path_a + path_b
                    #print("path: ", path)
                    return path        

            
            working_on_tree_a = True

def BiRRT_smoothing():
    ################################################################
    # TODO your code to implement the birrt algorithm with smoothing
    ################################################################
    path = BiRRT()
    #print("initial path: ", path)
    N = 100

    for _ in range(N):
        m, n = None, None
        while m == n:
            m = np.random.randint(0, len(path))
            n = np.random.randint(0, len(path))

        m_node = RRT_Node(path[m])
        n_node = RRT_Node(path[n])
        if m < n:
            flag = steer_to(m_node, n_node)
            if flag:      
                path = path[:m+1] + path[n:]
        else:
            flag = steer_to(n_node, m_node)
            if flag:
                path = path[:n+1] + path[m:]

    return path

def get_position_from_conf(conf):
    set_joint_positions(ur5, UR5_JOINT_INDICES, conf)
    return p.getLinkState(ur5, p.getNumJoints(ur5) - 1)[:2][0]
###############################################################################
#your implementation ends here

if __name__ == "__main__":
    args = get_args()

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    
	# place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    if args.birrt:
        if args.smoothing:
            # using birrt with smoothing
            #print("Here")
            path_conf = BiRRT_smoothing()
            #print("After smoothing: ", path_conf)
        else:
            # using birrt without smoothing
            path_conf = BiRRT()
            #print("After BiRRT: ", path_conf)
    else:
        # using rrt
        path_conf = RRT()
        #print(path_conf)
        #set_joint_positions

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        # execute the path
        while True:
            q_prev = None
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                print(p.getLinkState(ur5, p.getNumJoints(ur5)-1)[:2][0])
                draw_sphere_marker(position=p.getLinkState(ur5, p.getNumJoints(ur5)-1)[:2][0], radius=0.01,
                                   color=[0,1,0,1])
                
                if q_prev is not None:
                    draw_path = steer_to_get_path(q_prev, q)
                    for q_small in draw_path:
                        draw_sphere_marker(position=get_position_from_conf(q_small), radius=0.005, color=[1,1,0,0.5])
                q_prev = q
                time.sleep(0.5)
            