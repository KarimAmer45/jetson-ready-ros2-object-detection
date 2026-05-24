#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hint: read this script fully and carefully before you start coding...

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import heapq
import itertools


def plot_path(path, n_start, n_goal, M, save_fp=None):
    plt.matshow(M, cmap="gray")
    if path.shape[0] > 2:
        plt.plot(path[:, 1], path[:, 0], 'b')
    plt.plot(n_start[1], n_start[0], 'or')
    plt.plot(n_goal[1], n_goal[0], 'xg')
    if save_fp is None:
        plt.show()
    else:
        plt.savefig(save_fp)
    plt.close()


def is_valid(v, thr_free=0.9):
    """
    v : freespace value of the map
    thr_free : threshold of a cell on the map to be considered free
    """
    return bool(np.all(v > thr_free))


def _as_cell(cell):
    return tuple(int(v) for v in cell)


def _in_bounds(cell, M):
    row, col = cell
    return 0 <= row < M.shape[0] and 0 <= col < M.shape[1]


def _is_free(cell, M):
    return _in_bounds(cell, M) and is_valid(M[cell])


def _neighbors(cell, M):
    row, col = cell
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nxt = (row + d_row, col + d_col)
        if _is_free(nxt, M):
            yield nxt


def _reconstruct_path(parent, goal):
    path = []
    cell = goal
    while cell is not None:
        path.append(np.array(cell, dtype=int))
        cell = parent[cell]
    path.reverse()
    return path


def _heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def plan_path_uninformed(n_start, n_goal, M):
    """
    Exe 1, Q1. Implement uniform cost search algorithm here 
    You may use the above function is_valid to select only the cells that belong to the free space.
    Use the Node class below to keep track of the nodes in the tree
    Note, the comments here is just a very high-level and is not the complete algorithm. 
    However, based on this comments, you should know which algorithm we need you to implement. 
    Do not deviate too far from these, or you will end up with the wrong algorithm.
    """
    start = _as_cell(n_start)
    goal = _as_cell(n_goal)
    visited = np.zeros(M.shape[:2], dtype=np.uint8)

    if not _is_free(start, M) or not _is_free(goal, M):
        return [], visited

    queue = deque([start])
    parent = {start: None}
    visited[start] = 1

    while queue:
        cell = queue.popleft()
        if cell == goal:
            return _reconstruct_path(parent, goal), visited

        for nxt in _neighbors(cell, M):
            if visited[nxt]:
                continue
            parent[nxt] = cell
            visited[nxt] = 1
            queue.append(nxt)

    return [], visited


def plan_path_astar(n_start, n_goal, M):
    """
    Exe 2, Q2. Implement A* here
    Hint: extend from previous exercise.
          Try not to copy paste but you can add optional parameters to the old function
          You may want to git commit before you do this though... ;)
    """
    return _astar_search(n_start, n_goal, M, heuristic_weight=1.0)


def plan_path_fast(n_start, n_goal, M):
    """
    Exe 3, Q2. Implement you fastest runtime planner here
    """
    return _astar_search(n_start, n_goal, M, heuristic_weight=1.5)


def _astar_search(n_start, n_goal, M, heuristic_weight=1.0):
    start = _as_cell(n_start)
    goal = _as_cell(n_goal)
    visited = np.zeros(M.shape[:2], dtype=np.uint8)

    if not _is_free(start, M) or not _is_free(goal, M):
        return [], visited

    counter = itertools.count()
    frontier = []
    heapq.heappush(frontier, (heuristic_weight * _heuristic(start, goal), 0, next(counter), start))
    parent = {start: None}
    best_cost = {start: 0}

    while frontier:
        _, cost, _, cell = heapq.heappop(frontier)
        if visited[cell]:
            continue
        visited[cell] = 1

        if cell == goal:
            return _reconstruct_path(parent, goal), visited

        for nxt in _neighbors(cell, M):
            new_cost = cost + 1
            if new_cost >= best_cost.get(nxt, np.inf):
                continue
            best_cost[nxt] = new_cost
            parent[nxt] = cell
            priority = new_cost + heuristic_weight * _heuristic(nxt, goal)
            heapq.heappush(frontier, (priority, new_cost, next(counter), nxt))

    return [], visited

class Node:
    """Use this class to help with keeping track of the paths
    """
    def __init__(self, parent = None, cell = None):
        # parent is the node from which you reach the current one
        self.parent = parent
        # cell is the position of the node 
        self.cell = cell 
        # values for the cost functions
        self.g = 0
        self.h = 0
        self.f = 0
        
    def __str__(self):
        return str(self.cell)
    
    def __repr__(self):
        return str(self)
