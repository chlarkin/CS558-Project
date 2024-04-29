List of Files and their purpose

data_gen.py
    File used to generate data for neural net training. (Will create the data and save it)

model_test.py
    File used to evaluate trained models.

UR5_RRTstar_classical.py
    Implements RRT* on UR5 and environment. Outputs the time spent in collision checker, and cost of path. Will rerun until it finds a path (1000 iterations per run)

UR5_RRTstar_hybrid.py
    Implements RRT* with hybrid collision checker on UR5 and environment. Outputs the time spent in collision checker, and cost of path. Will rerun until it finds a path (1000 iterations per run)

UR5_RRTstar_NN.py
    Implements RRT* with trained collision checker on UR5 and environment. Outputs the time spent in collision checker, and cost of path. Will rerun until it finds a path (1000 iterations per run)

Training.py
    Trains neural net and saves it to specified location

visualizer.py
    Allows for visualization of specific joints angles and environments. First 2 lines of file are the only thing to be edited