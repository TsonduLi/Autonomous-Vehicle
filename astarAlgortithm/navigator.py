#!/usr/bin/env python3
import numpy as np
import typing as T
import math
import rclpy
import scipy
from scipy.interpolate import splev
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from rclpy.node import Node
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D

from P1_astar import DetOccupancyGrid2D, AStar
from utils import generate_planning_problem

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from utils import plot_line_segments



class HW2Navigator(BaseNavigator):
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("hw2_navigator")
        self.Kp = 2.0

        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.
        self.v_desired = 0.15

        self.coeffs = np.zeros(8) # Polynomial coefficients for x(t) and y(t) as
                                  # returned by the differential flatness code

        

    def compute_heading_control(self,
        state: TurtleBotState,
        goal: TurtleBotState) -> TurtleBotControl:
        """ Compute control given current robot state and goal state

        Args:
            state (TurtleBotState): current robot state
            goal (TurtleBotState): current goal state

        Returns:
            TurtleBotControl: control command
        """

        control_command = TurtleBotControl()
        th_state = state.theta
        th_goal = goal.theta
        th_err = wrap_angle(th_goal - th_state)
        ang_velo = self.Kp*th_err
        control_command.omega = ang_velo
        return control_command
    
    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:
        """ Compute control target using a trajectory tracking controller

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """

        # current states
        dt = t - self.t_prev
        # x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)
        th = state.theta
        x = state.x
        y = state.y

        # desired states
        x_d = float(splev(t, plan.path_x_spline, der=0))
        xd_d = float(splev(t, plan.path_x_spline, der=1))
        xdd_d = float(splev(t, plan.path_x_spline, der=2))
        y_d = float(splev(t, plan.path_y_spline, der=0))
        yd_d = float(splev(t, plan.path_y_spline, der=1))
        ydd_d = float(splev(t, plan.path_y_spline, der=2))

        # delta states
        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)

        # control input
        u1 = xdd_d + self.kpx*(x_d - x) + self.kdx*(xd_d - xd)
        u2 = ydd_d + self.kpy*(y_d - y) + self.kdy*(yd_d - yd)
        Vd = u1*np.cos(th) + u2*np.sin(th)
        V = self.V_prev + dt*Vd
        om = (u2*np.cos(th) - u1*np.sin(th)) / V

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        control_command = TurtleBotControl()
        control_command.omega = om
        control_command.v = V
        return control_command
    
    
    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting

        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """

        # x_init = (state.x, state.y)
        # x_goal = (goal.x, goal.y)
        x_init = np.array([state.x, state.y])
        x_goal = np.array([goal.x, goal.y])
        lb = (state.x - horizon, state.y - horizon)
        hb = (state.x + horizon, state.y + horizon)

        astar = AStar(lb, hb, x_init, x_goal, occupancy, resolution)
        if not astar.solve():
            print("No path found")
            return None
        else:
            path = np.asarray(astar.path)

        if np.shape(path)[0] < 4:
            print("Path found, but length < 4")
            return None
        
        # new traj, reset t & velo
        self.t_prev = 0
        self.V_prev = 0 

        v_desired = self.v_desired
        # traj = TrajectoryPlan()
        ts_n = np.shape(path)[0]
        ts = np.zeros(ts_n)
        for i in range(ts_n-1):
            ts[i+1] = np.linalg.norm(path[i+1] - path[i]) / v_desired 
            ts[i+1] = ts[i+1] + ts[i]
        print(ts)
        print(path[: ,0])
        path_x_spline = scipy.interpolate.splrep(ts, path[: ,0], k=3)
        path_y_spline = scipy.interpolate.splrep(ts, path[: ,1], k=3)
        
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = HW2Navigator()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits