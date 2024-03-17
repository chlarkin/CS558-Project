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
parent_node = RRT_Node((1,2,3))

node = RRT_Node((0,0,0))
node.set_parent(parent_node)
parent_node.add_child(node)
# print(node.conf)

node.cost = 1
# print(node.cost)

node1 = RRT_Node((1,1,1))
node2 = RRT_Node((2,2,2))
node3 = RRT_Node((3,3,3))
# node4 = RRT_Node((4,4,4))
node5 = RRT_Node((5,5,5))
node6 = RRT_Node((6,6,6))
node7 = RRT_Node((7,7,7))
node8 = RRT_Node((8,8,8))
node9 = RRT_Node((9,9,9))
node10 = RRT_Node((10,10,10))

node.add_child(node1)
node1.set_parent(node)
node.add_child(node2)
node2.set_parent(node)

node1.add_child(node3)
node3.set_parent(node1)
node1.add_child(node5)
node5.set_parent(node1)

node3.add_child(node7)
node7.set_parent(node3)

node2.add_child(node6)
node6.set_parent(node2)
node2.add_child(node8)
node8.set_parent(node2)

node8.add_child(node9)
node9.set_parent(node8)
node8.add_child(node10)
node10.set_parent(node8)

#Below is required code
empty = []
list = []
current_node = node.child[0]
list.append(current_node.conf)
current_node.cost = current_node.parent.cost + 1
level = 0

print(node.child)
if node1 in node.child:
    node.child.remove(node1)
elif node3 in node.child:
    print("no")
print(node.child)

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


