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


node = RRT_Node((1,1,1))

print(node.conf)

node.cost = 10
print(node.cost)

node2 = RRT_Node((2,2,2))
node3 = RRT_Node((3,3,3))

node.add_child(node2)
node.add_child(node3)

print(node.child)