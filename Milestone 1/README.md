List of Files and their purpose

Collision Utils.py
    collision checker for UR5

data_gen.py
    File used to generate data for neural net training. (Will create the data and save it)

data_testing.py
    Used for testing syntax, etc. can be ignored
testing_file.py
    Used for testing syntax, etc. can be ignored

UR5_RRTstar.py
    Implements RRT* on UR5 and environment. Outputs the time spent in collision checker, and cost of path. Will rerun until it finds a path (1000 iterations per run)

env_generator.py
    Used to randomly place the obstacles. Manually used to create all environments

visualizer.py
    Allows for visualization of specific joints angles and environments. First 2 lines of file are the only thing to be edited