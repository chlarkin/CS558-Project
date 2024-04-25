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

class run_time:
    def __init__(self, t):
        self.t = t
    def add_t(self, t):
        self.t += t

class RRT_Node:
    def __init__(self, conf):
        self.conf = conf
        self.child = []

    def set_parent(self, parent):
        if self.conf != parent.conf:
            self.parent = parent

    def add_child(self, child):
        if child not in self.child:
            self.child.append(child)

    def set_cost(self, cost):
        self.cost = cost

def sample_conf():
    if random.randint(0, 100) > 98: #Goal config 2% of the time
        return goal_conf #Goal
    else:
        sample_joint1 = np.random.uniform(-2*math.pi,2*math.pi)
        sample_joint2 = np.random.uniform(-2*math.pi,2*math.pi)
        sample_joint3 = np.random.uniform(-math.pi,math.pi)
        
        rand_conf = (sample_joint1, sample_joint2, sample_joint3)    
        return rand_conf #Random
   
def find_nearest(rand_node, node_list):
    q_nearest = node_list[0]
    for i in node_list:
        if (math.dist(i.conf, rand_node.conf) < math.dist(q_nearest.conf, rand_node.conf)):
            q_nearest = i
    
    return q_nearest

def steer_to(rand_node, nearest_node, collision_runtime): 
    collision_flag = 0
    step_size = 0.2
   
    diff = np.array(rand_node.conf) - np.array(nearest_node.conf)
    cost = np.linalg.norm(diff)
    if cost == 0:
        return 0, cost
    num_step = abs(math.ceil(math.dist(rand_node.conf, nearest_node.conf) / step_size)) #divide distance by step size, round up
    stepper = diff / cost * step_size
    v = np.zeros([num_step+1, 3])
    v[num_step, :] = rand_node.conf
    
    for i in range(0, num_step):
        v[i,:] = nearest_node.conf + (stepper * i)
    starttime = time.time()
    for q in v:
        
        if (collision_fn(q)): ### To be replaced with NN
            endtime = time.time()
            collision_runtime.add_t(endtime-starttime)
            collision_flag = 1 #There is collision
            return collision_flag, cost
    
    endtime = time.time()
    collision_runtime.add_t(endtime-starttime)    
    
    return collision_flag, cost

def near(q_rand, T):

    # Use this value of gamma
    GAMMA = 5

    # your code here
    v = len(T)
    n = 3
    Xnear = []

    radius = GAMMA * (math.log(abs(v)) / abs(v)) ** (1/n) #calculating radius
    # print(radius)

    for node in T:
        norm_node = np.linalg.norm(np.array(node.conf))
        norm_q_rand = np.linalg.norm(np.array(q_rand.conf))
        if ((norm_node - norm_q_rand)<= radius):
            Xnear.append(node) 

    return Xnear #Returns nodes within radius

def choose_parent(rand, Xnear, collision_runtime):
        if len(Xnear) == 0: #No nodes nearby
            return None
        
        min_node = None
        cost_list = []
        ind_list = []
        for node in Xnear:
            flag, new_to_near_cost = steer_to(rand, node, collision_runtime) #first check if you can steer to rand node
            if flag != 1:
                cost_list.append(node.cost + new_to_near_cost) #cost of current node + cost of newNode to current node
                ind_list.append(node) #index of node corresponding to cost calculated in line above

        if len(cost_list) == 0:
            return None #No collision free paths found

        min_node = ind_list[cost_list.index(min(cost_list))]
    
        return min_node

def rewire(Xnear, rand_node, collision_runtime):
    if len(Xnear) == 0:
        return
    for node in Xnear:
        current_cost = node.cost
        flag, new_cost = steer_to(rand_node, node, collision_runtime)

        if flag != 1:
            new_cost += rand_node.cost
            if new_cost < current_cost:
                node.parent.child.remove(node) #Old parent no longer has this as a child
                node.set_parent(rand_node) #New parent assigned
                rand_node.add_child(node) #Child added
                node.cost = new_cost
                if rand_node.conf != goal_conf:
                    rand_node.add_child(node)
                tree_traversal(node)
    return

def tree_traversal(node):
    #Updates cost of all decendants of a given node
    empty = []
    # list = []
    if node.child == empty:
        return
    current_node = node.child[0]
    # list.append(current_node)
    current_node.cost = current_node.parent.cost + np.linalg.norm(np.array(current_node.parent.conf) - np.array(current_node.conf))
    level = 1
    while current_node != node:
        # print(level)
        # print(current_node.conf)
        # print(current_node.child)
        # print(current_node.parent.conf)
        # print(current_node.parent.child)
        if level < 0:
            print(current_node.conf)
            break
        #Move down
        if current_node.child != empty:
            # print("down")
            level += 1
            current_node = current_node.child[0]
            current_node.cost = current_node.parent.cost + np.linalg.norm(np.array(current_node.parent.conf) - np.array(current_node.conf))
            # list.append(current_node)

        #Move right
        elif current_node.parent.child.index(current_node) + 1 < len(current_node.parent.child):
            # print("right")
            current_node = current_node.parent.child[current_node.parent.child.index(current_node) + 1]
            current_node.cost = current_node.parent.cost + np.linalg.norm(np.array(current_node.parent.conf) - np.array(current_node.conf))
            # list.append(current_node)
        
        #Move up and right or until node
        else: 
            while (current_node.parent.child.index(current_node) + 1 >= len(current_node.parent.child)) and (current_node.conf != node.conf):
                current_node = current_node.parent
                level -= 1
                # print("up")
            if current_node.conf == node.conf:
                pass #end loop when it reaches while
            else:
                # print("right (up)")
                current_node = current_node.parent.child[current_node.parent.child.index(current_node) + 1]
                current_node.cost = current_node.parent.cost + np.linalg.norm(np.array(current_node.parent.conf) - np.array(current_node.conf))
                # list.append(current_node)
    return

def RRT_star():
    #Implement full RRT Star algorithm
    max_iterations = 1000
    num_iterations = 0
    goal_node = RRT_Node(goal_conf)
    collision_runtime = run_time(0)
    goal_flag = False
    assist_nodes = []

    #Initialize Tree
    T = []
    T.append(RRT_Node(start_conf))
    T[0].cost = 0 #Start goal

    for i in range(1, max_iterations):
        num_iterations += 1
        #Sample Configuration
        q_rand = sample_conf() #generate random configuration
        q_rand = RRT_Node(q_rand) #make it a node
        if q_rand.conf == goal_node.conf:
            q_rand = goal_node #This is for checking if goal node is in T later
        
        #Find Nearest Node to Random Node
        q_nearest = find_nearest(q_rand, T) #find nearest node to random node
        
        #Collision Free Path?
        collision_flag, cost = steer_to(q_rand, q_nearest, collision_runtime)
        
        #If path is valid
        if (collision_flag != 1):
            
            #Find near parents
            Xnear = near(q_rand, T) #nodes within radius
            min_parent = choose_parent(q_rand, Xnear, collision_runtime)
            
            if min_parent is not None:
                q_rand.set_parent(min_parent)
                q_rand.cost = min_parent.cost + np.linalg.norm(np.array(q_rand.conf) - np.array(min_parent.conf))
            else:
                q_rand.set_parent(q_nearest)
                q_rand.cost = q_nearest.cost + cost
            
            q_rand.parent.add_child(q_rand)
            
            #Add to tree if not already in tree (Would only happen if its goal node)
            if q_rand not in T:
                T.append(q_rand)
            
            
            if q_rand.conf == goal_conf:
                print(num_iterations)
                goal_flag = True
                assist_nodes.append(q_rand.parent) 
                
            #Rewire Tree
            rewire(Xnear, q_rand, collision_runtime)
    total_cost = []
    if goal_flag:
        for assist in assist_nodes:
            assist_cost = assist.cost
            goal_cost = np.linalg.norm(np.array(goal_node.conf) - np.array(assist.conf))
            total_cost.append(assist_cost + goal_cost)
            # add code to sort through this list
        min_assist = assist_nodes[total_cost.index(min(total_cost))]
        goal_node.set_parent(min_assist)
    else:
        print(f"Goal not found in {max_iterations} iterations")
        path_conf = []
        return None
    
    #Creating Path Config
    path_conf = []
    q_path = goal_node
    counter = 0
    while (q_path != T[0]) and (counter < 20):
        counter += 1
        path_conf.append(q_path.conf)
        q_path = q_path.parent

    if counter > 15:
        print(path_conf)
    path_conf.append(T[0].conf)
    path_conf.reverse()
    # print("Number of iterations (RRT):")
    # print(num_iterations)
    path_cost = 0
    for i in range(len(path_conf)-1):
        diff = np.array(path_conf[i+1]) - np.array(path_conf[i])
        cost = np.linalg.norm(diff)
        path_cost += cost
    print(f'The cost of this path is: {path_cost}')
    print(f'The runtime for the collision checker is {collision_runtime.t}')
    path_conf = path_smoothing(path_conf)
    return path_conf

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
    # plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [obstacle1, obstacle2]
    
    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf =  (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    #Collision Runtime
    
    
	# place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    path_conf = RRT_star()

    while path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
        path_conf = RRT_star()
    else:

        # execute the path
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(.051)
            time.sleep(0.5)