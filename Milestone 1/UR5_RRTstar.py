from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import math
import random
import os
import sys


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

class RRT_Node:
    def __init__(self, conf):
        self.conf = conf
        self.child = []

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.child.append(child)

    def set_cost(self, cost):
        self.cost = cost

def sample_conf():
    if random.randint(0, 100) > 90: #Goal config 10% of the time
        return goal_conf, 1 #Goal
    else:
        sample_joint1 = np.random.uniform(-2*math.pi,2*math.pi)
        sample_joint2 = np.random.uniform(-2*math.pi,2*math.pi)
        sample_joint3 = np.random.uniform(-math.pi,math.pi)
        
        rand_conf = (sample_joint1, sample_joint2, sample_joint3)    
        return rand_conf, 0 #Random
   
def find_nearest(rand_node, node_list):
    q_nearest = node_list[0]
    for i in node_list:
        if (math.dist(i.conf, rand_node.conf) < math.dist(q_nearest.conf, rand_node.conf)):
            q_nearest = i
    
    return q_nearest

def steer_to(rand_node, nearest_node): 
    collision_flag = 0
    step_size = 0.05
    
    diff = np.array(rand_node.conf) - np.array(nearest_node.conf)
    cost = np.linalg.norm(diff)
    num_step = abs(math.ceil(math.dist(rand_node.conf, nearest_node.conf) / step_size)) #divide distance by step size, round up
    stepper = diff / np.linalg.norm(diff) * step_size
    v = np.zeros([num_step+1, 3])
    v[num_step, :] = rand_node.conf
    
    for i in range(0, num_step):
        v[i,:] = nearest_node.conf + (stepper * i)
    
    for q in v:
        if (collision_fn(q)): ### To be replaced with NN
            print("here")
            collision_flag = 1 #There is collision
            break
    
    return collision_flag, cost

def near(q_rand, T):

    # Use this value of gamma
    GAMMA = 50

    # your code here
    v = len(T)
    n = 3
    Xnear = []

    radius = GAMMA * (math.log(abs(v)) / abs(v)) ** (1/n) #calculating radius

    for node in T:
        norm_node = np.linalg.norm(np.array(node.conf))
        norm_q_rand = np.linalg.norm(np.array(q_rand.conf))
        if ((norm_node - norm_q_rand)<= radius):
            Xnear.append(node) #Returns nodes within radius

    return Xnear

def choose_parent(rand, Xnear):
        if len(Xnear) == 0: #No nodes nearby
            return None
        
        min_node = None
        cost_list = []
        ind_list = []
        for node in Xnear:
            flag, new_to_near_cost = steer_to(rand, node) #first check if you can steer to rand node
            if flag:
                cost_list.append(node.cost + new_to_near_cost) #cost of current node + cost of newNode to current node
                ind_list.append(node) #index of node corresponding to cost calculated in line above

        if len(cost_list) == 0:
            return None #No collision free paths found

        min_node = ind_list[cost_list.index(min(cost_list))]
    
        return min_node

def rewire(Xnear, new_node):
    if Xnear == None:
        return
    for node in Xnear:
        current_cost = node.cost
        flag, new_cost = steer_to(new_node, node)

        if flag != 1:
            new_cost += new_node.cost
            if new_cost < current_cost:
                node.parent = new_node
                node.cost = new_cost
                new_node.add_child(node)
                #Update decendants cost
                

    return


#Self-Made Fucntion for testing ###################
def path_smoothing(path_conf):
    step_size = 0.05
    path_conf_stepped = []
    for i in range(1, len(path_conf)):
        path_conf_stepped.append(path_conf[i-1])
        diff = np.array(path_conf[i]) - np.array(path_conf[i-1])
        num_step = abs(math.ceil(math.dist(path_conf[i], path_conf[i-1]) / step_size)) #divide distance by step size, round up
        stepper = diff / np.linalg.norm(diff) * step_size
    
        for j in range(0, num_step):
            path_conf_stepped.append(path_conf[i-1] + (stepper * j))
    path_conf_stepped.append(path_conf[len(path_conf)-1])
    return path_conf_stepped 
####################################################

def RRT_star():
    #Implement full RRT Star algorithm
    max_iterations = 3000
    count = 0
    num_iterations = 1
    goal_node = RRT_Node(goal_conf)
    
    #Initialize Tree
    T = []
    T.append(RRT_Node(start_conf))
    T[0].cost = 0 #Start goal

    for i in range(1, max_iterations):
        num_iterations += 1
        #Sample Configuration
        q_rand, goal_flag = sample_conf() #generate random configuration
        q_rand = RRT_Node(q_rand) #make it a node
        
        #Find Nearest Node to Random Node
        q_nearest = find_nearest(q_rand, T) #find nearest node to random node
        
        #Collision Free Path?
        collision_flag, cost = steer_to(q_rand, q_nearest)
        q_rand.cost = cost

        #If path is valid
        if (collision_flag != 1):
            count += 1
            
            #Find near parents
            Xnear = near(q_rand, T)
            min_parent = choose_parent(q_rand, Xnear)
            if min_parent is not None:
                q_rand.parent = min_parent
                q_rand.cost = min_parent.cost + np.linalg.norm(np.array(q_rand.conf) - np.array(min_parent.conf))
                min_parent.add_child(q_rand)
            else:
                q_rand.parent = q_nearest
                q_rand.cost = q_nearest.cost + cost
                q_nearest.add_child(q_rand)
            
            #Add to tree
            T.append(q_rand)
            
            #Rewire Tree


        else: #no path from rand node to nearest node
            goal_flag = 0
        if(goal_flag):
            if steer_to(q_rand, goal_node) != 1:
                T.append(goal_node)
                goal_node.parent = q_rand
                break
            else: #no path from rand_node to goal node (rand node is still added to tree with nearest node)
                goal_flag = 0
    
    if (num_iterations == max_iterations):
        path_conf = []
        return None
    
    #Creating Path Config
    path_conf = []
    q_path = T[len(T)-1]

    while (q_path.conf != T[0].conf):
        path_conf.append(q_path.conf)
        q_path = q_path.parent
    path_conf.append(T[0].conf)
    path_conf.reverse()
    # print("Number of iterations (RRT):")
    # print(num_iterations)
    path_conf = path_smoothing(path_conf)
    return path_conf

###############################################################################
#your implementation ends here

if __name__ == "__main__":
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
    goal_conf =  (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
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

    path_conf = RRT_star()

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:

        # execute the path
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(.051)
            time.sleep(0.5)