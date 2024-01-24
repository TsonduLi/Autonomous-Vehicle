#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from visualize import plot_line_segments
from asl_tb3_lib.grids import StochOccupancyGrid2D
from typing import Tuple
from collections import defaultdict
from asl_tb3_lib.navigation import TrajectoryPlan
from scipy.interpolate import splrep

class AStar(object):
    def __init__(
        self,
        statespace_lo: Tuple,
        statespace_hi: Tuple,
        x_init: Tuple,
        x_goal: Tuple,
        occupancy: StochOccupancyGrid2D,
        resolution: float = 1.
    ):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = defaultdict(lambda: np.inf)    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init, self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        return self.occupancy.is_free(x)

    def in_map(self, x):
        is_in_map = x[0] >= self.statespace_lo[0] \
            and x[0] <= self.statespace_hi[0] \
            and x[1] >= self.statespace_lo[1] \
            and x[1] <= self.statespace_hi[1]
        return is_in_map

    def distance(self, x1, x2):
        return np.linalg.norm(np.array(x1) - np.array(x2), ord=2)

    def snap_to_grid(self, x):
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        a_range = [x[0] - self.resolution, x[0], x[0] + self.resolution]
        b_range = [x[1] - self.resolution, x[1], x[1] + self.resolution]
        neighbors = [
            (a, b) for a in a_range for b in b_range \
            if (a, b) != x and self.is_free(np.array([a, b])) and self.in_map((a, b))
        ]
        return neighbors

    def find_best_est_cost_through(self):
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        self.path = np.array(list(reversed(path)))

    def plot_path(self, fig_num=0, show_init_label=True):
        if not self.path:
            return
        self.occupancy.plot(fig_num)
        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        x_cur = self.x_init
        while len(self.open_set) > 0:
            neighbors = self.get_neighbors(x_cur)
            for x_n in neighbors:
                if x_n in self.closed_set:
                    continue
                tmp_cost = self.cost_to_arrive[x_cur] + self.distance(x_cur, x_n)
                if x_n not in self.open_set:
                    self.open_set.add(x_n)
                elif tmp_cost > self.cost_to_arrive[x_n]:
                    continue
                self.came_from[x_n] = x_cur
                self.cost_to_arrive[x_n] = tmp_cost
                self.est_cost_through[x_n] = tmp_cost + self.distance(x_n, self.x_goal)

            x_cur = self.find_best_est_cost_through()
            if x_cur == self.x_goal:
                self.reconstruct_path()
                return True
            self.open_set.remove(x_cur)
            self.closed_set.add(x_cur)
        return False

def compute_smooth_plan(path, k, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
    # Ensure path is a numpy array
    path = np.asarray(path)
    xdiff = np.abs(np.ediff1d(path[:, 0], to_begin=0))
    ydiff = np.abs(np.ediff1d(path[:, 1], to_begin=0))
    ts = np.cumsum((xdiff + ydiff) / v_desired)
    path_x_spline = splrep(ts, path[:, 0], k=k, s=spline_alpha)
    path_y_spline = splrep(ts, path[:, 1], k=k, s=spline_alpha)
    path_plan = TrajectoryPlan(
        path=path,
        path_x_spline=path_x_spline,
        path_y_spline=path_y_spline,
        duration=ts[-1]
    )
    return path_plan
    