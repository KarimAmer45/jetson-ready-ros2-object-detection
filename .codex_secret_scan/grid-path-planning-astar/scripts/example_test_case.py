"""Script to run and visualise path planned.
Use this script to check that your code works. 
You can uncomment the code based on which function you want to test.
I highly recommend you write your own test functions as well.

BIG HINT: we use a very similar script to evaluate your code. 
So make sure that your code works well in this script!
"""

import time 

import numpy as np 
import matplotlib.image as mpimg

import assignment_1_astar as utils

def print_stats(path, visited, delta_time):
    print(f"path length: {len(path)}")
    print(f"num. nodes explored: {(visited!=0).sum()}")
    print(f"time taken (s): {delta_time}")

def test_1(planner, planner_str):
    M = np.ones([15,15])
    M[3,2:10] = 0
    M[5,7:10] = 0
    M[5:10,2:10] = 0
    M[5:15,5] = 0

    x_start = np.array([2,2])
    x_goal = np.array([14,14])

    start_time = time.time()
    path, visited = planner(x_start, x_goal, M)
    end_time = time.time()
    print_stats(path, visited, end_time-start_time)
    utils.plot_path(np.asarray(path), x_start, x_goal, M, save_fp = f"{planner_str}_test_1.png")


def test_2(planner, planner_str):
    """test using a map read from an image file
    """
    M = mpimg.imread('../map.png')

    x_start = np.array([500, 410])
    x_goal = np.array([495, 35])

    start_time = time.time()
    path, visited = planner(x_start, x_goal, M)
    end_time = time.time()

    print_stats(path, visited, end_time-start_time)
    utils.plot_path(np.asarray(path), x_start, x_goal, M, save_fp = f"{planner_str}_test_2.png")


def test_3(planner, planner_str):
    """generate your own tests.
    e.g., test on a larger scale using 100 random maps and inits
    """
    pass

    
if __name__ == '__main__':
    print("uninformed: ")
    test_1(utils.plan_path_uninformed, "uninformed")
    test_2(utils.plan_path_uninformed, "uninformed")

    print("A*: ")
    test_1(utils.plan_path_astar, "astar")
    test_2(utils.plan_path_astar, "astar")

    print("Fast: ")
    test_1(utils.plan_path_fast, "fast")
    test_2(utils.plan_path_fast, "fast")
