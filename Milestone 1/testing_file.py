import numpy as np
envs = np.array([[[-0.48606829713512445, -0.4742208703810389, 0.2026271262856553],       [0.4891267201053682, 0.07522346831553983, 0.5050351665101648]],         #Env 0
                    [[0.3002731922279034, -0.31160217067115104, 0.39323507803345176],    [-0.3985446254433046, 0.041775612004733564, 0.5793198619125965]],       #Env 1
                    [[0.146045675104168, -0.3979542571960254, 0.4675609435444279],       [0.46211068055476157, -0.08911039520897335, 0.5462541154340267]],       #Env 2
                    [[-0.3749089906479599, -0.010920546606923653, 0.6548380413171766],   [-0.4717427258801794, -0.34305068452267706, 0.38673039718973085]],      #Env 3
                    [[-0.39444105916439776, -0.27913277812558424, 0.0861344814155049],   [-0.057691336927091275, 0.3933244574605025, 0.6340804234643873]],       #Env 4
                    [[-0.3937234042986071, -0.3329806784419763, 0.48294830674568257],    [-0.14931602505734265, -0.35327940059799556, 0.7467517990261291]],      #Env 5
                    [[0.24619394897823332, -0.46634978538240346, 0.6036769084407155],    [-0.188771305474139, -0.48355682893226926, 0.6502166199026057]],        #Env 6
                    [[0.30148022031916966, 0.48695418445995076, 0.2535818443385577],     [0.004332701486908563, 0.46267510139160917, 0.08813582725439109]],      #Env 7
                    [[-0.41415525113482976, 0.06557705887582732, 0.3162571017012827],    [-0.32518020001298464, 0.29626074239715006, 0.28787093307336853]],      #Env 8
                    [[0.423925580570622, -0.4310000806945503, 0.45444298410941264],      [-0.46976501809520543, 0.16608826151479195, 0.6178054289271393]],       #Env 9
                    [[-0.36043810414437105, -0.2691389029682508, 0.686207439235861],     [-0.16202069573457856, 0.08022645314133292, 0.26920070814356534]],      #Env 10
                    [[0.4566458746084171, -0.36699231220946027, 0.048405981110216434],   [0.3604667357065606, -0.4480200275276519, 0.7438370081125503]],         #Env 11
                    [[-0.3496936173579349, -0.08822492422114869, 0.3371807411354862],    [0.34053619785062583, -0.13010504174460635, 0.4657493670182545]],       #Env 12
                    [[0.00869058242837939, 0.3958734299236678, 0.38024399733000114],     [-0.25508255027799, 0.005429310422922273, 0.26672253744663454]],        #Env 13
                    [[0.14389874257136936, 0.4196948341180933, 0.00806432747263408],     [0.12006162599651782, 0.4011229253195293, 0.4421590244474948]],         #Env 14
                    [[-0.0634793598803739, -0.44795587631399514, 0.44465436231129235],   [-0.44517936107569245, 0.41965381346445696, 0.5537185542645229]],       #Env 15
                    [[0.19740309163839387, -0.2062963110733218, 0.14469312016255007],    [-0.07813323529404825, -0.3649518721772905, 0.2986866388921594]],       #Env 16
                    [[0.20200492453930008, -0.09701915256973404, 0.3224559182919058],    [0.4899902911771462, -0.08504320404400856, 0.36700568222796737]],       #Env 17
                    [[-0.16133839804507633, -0.003810313573494706, 0.35169426584596664], [-0.14816725470594383, 0.4594167822909051, 0.621533084927139]],         #Env 18
                    [[-0.4955572767711125, 0.16964750789664074, 0.6974576123145267],     [0.2536177002826495, -0.2736101732601708, 0.7298064484091686]],         #Env 19
                    [[-0.41244751782205946, 0.29121829054968384, 0.5880517816452684],    [-0.3806322632383895, 0.4543174691105484, 0.15959840349033577]],        #Env 20
                    ])

obstacle_positions = envs[10]
print(obstacle_positions)

# class RRT_Node:
#     def __init__(self, conf):
#         self.conf = conf
#         self.child = []

#     def set_parent(self, parent):
#         self.parent = parent

#     def add_child(self, child):
#         self.child.append(child)

#     def set_cost(self, cost):
#         self.cost = cost
# parent_node = RRT_Node((1,2,3))

# node = RRT_Node((0,0,0))
# node.set_parent(parent_node)
# parent_node.add_child(node)
# # print(node.conf)

# node.cost = 1
# # print(node.cost)

# node1 = RRT_Node((1,1,1))
# node2 = RRT_Node((2,2,2))
# node3 = RRT_Node((3,3,3))
# # node4 = RRT_Node((4,4,4))
# node5 = RRT_Node((5,5,5))
# node6 = RRT_Node((6,6,6))
# node7 = RRT_Node((7,7,7))
# node8 = RRT_Node((8,8,8))
# node9 = RRT_Node((9,9,9))
# node10 = RRT_Node((10,10,10))

# node.add_child(node1)
# node1.set_parent(node)
# node.add_child(node2)
# node2.set_parent(node)

# node1.add_child(node3)
# node3.set_parent(node1)
# node1.add_child(node5)
# node5.set_parent(node1)

# node3.add_child(node7)
# node7.set_parent(node3)

# node2.add_child(node6)
# node6.set_parent(node2)
# node2.add_child(node8)
# node8.set_parent(node2)

# node8.add_child(node9)
# node9.set_parent(node8)
# node8.add_child(node10)
# node10.set_parent(node8)

# #Below is required code
# empty = []
# list = []
# current_node = node.child[0]
# list.append(current_node.conf)
# current_node.cost = current_node.parent.cost + 1
# level = 0

# print(node.child)
# if node1 in node.child:
#     node.child.remove(node1)
# elif node3 in node.child:
#     print("no")
# print(node.child)

# while current_node != node:
#     # print(current_node.conf)
#     print(level)
#     #Move down
#     if current_node.child != empty:
#         level += 1
#         # print("down")
#         current_node = current_node.child[0]
#         current_node.cost = current_node.parent.cost + 1
#         list.append(current_node.conf)

#     #Move right
#     elif current_node.parent.child.index(current_node) + 1 < len(current_node.parent.child):
#         # print("right")
#         current_node = current_node.parent.child[current_node.parent.child.index(current_node) + 1]
#         current_node.cost = current_node.parent.cost + 1
#         list.append(current_node.conf)
    
#     #Move up and right or until node
#     else: 
#         while (current_node.parent.child.index(current_node) + 1 >= len(current_node.parent.child)) and (current_node != node):
#             current_node = current_node.parent
#             level -= 1
#             # print("up")
#             # print(current_node.conf)
            
#         if current_node == node:
#             pass #end loop when it reaches while
#         else:
#             # print("right (up)")
#             current_node = current_node.parent.child[current_node.parent.child.index(current_node) + 1]
#             # print(current_node.conf)
#             current_node.cost = current_node.parent.cost + 1
#             list.append(current_node.conf)


# print(list)


