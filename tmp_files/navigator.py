#!/usr/bin/env python3

import rclpy
import numpy as np
from scipy.interpolate import splev, splrep
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_lib.navigation import TrajectoryPlan, BaseNavigator
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from astar import AStar, compute_smooth_plan
from asl_tb3_lib.navigation import TrajectoryPlan
import numpy as np

class MyNavigator(BaseNavigator):
    def __init__(self, **kwargs) -> None:
        super().__init__("controller", **kwargs)
        self.gains_lookup = {}
        self.V_PREV_THRES = kwargs.get('V_PREV_THRES', 0.01)
        self.kpx = kwargs.get('kpx', 2.)
        self.kpy = kwargs.get('kpy', 2.)
        self.kdx = kwargs.get('kdx', 2.)
        self.kdy = kwargs.get('kdy', 2.)
        self.V_prev = kwargs.get('v_init', 0.)
        self.t_prev = kwargs.get('t_init', 0.)
        self.kp = kwargs.get('kp', 2.0)
        self.v_desired = kwargs.get('v_desired', 0.15)
        self.spline_alpha = kwargs.get('spline_alpha', 0.05)
        self.map_width = kwargs.get('map_width', 10)
        self.map_height = kwargs.get('map_height', 10)
        self.kwargs = kwargs
    
    def reset(self):
        self.V_prev = self.kwargs.get('v_init', 0.)
        self.om_prev = self.kwargs.get('om_init', 0.)
        self.t_prev = self.kwargs.get('t_init', 0.)
        self.x_prev = self.kwargs.get('x_init', 0.)
        self.y_prev = self.kwargs.get('y_init', 0.)

    def compute_heading_control(
        self,
        state: TurtleBotState,
        goal: TurtleBotState
    ) -> TurtleBotControl:
        heading_err = wrap_angle(goal.theta - state.theta)
        om = self.kp * heading_err
        ctrl_out = TurtleBotControl()
        ctrl_out.omega = om
        return ctrl_out

    def get_desired_state(self, t):
        x_d = splev(t, self.plan.path_x_spline, der=0)
        y_d = splev(t, self.plan.path_y_spline, der=0)
        xd_d = splev(t, self.plan.path_x_spline, der=1)
        yd_d = splev(t, self.plan.path_y_spline, der=1)
        xdd_d = splev(t, self.plan.path_x_spline, der=2)
        ydd_d = splev(t, self.plan.path_y_spline, der=2)
        return x_d, y_d, xd_d, yd_d, xdd_d, ydd_d

    def compute_trajectory_tracking_control(
        self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float
    ) -> TurtleBotControl:
        self.plan=plan
        x, y, th = state.x, state.y, state.theta
        dt = t - self.t_prev
        x_d, y_d, xd_d, yd_d, xdd_d, ydd_d = self.get_desired_state(t)

        ########## Code starts here ##########
        self.V_prev = max(self.V_prev, self.V_PREV_THRES)
        xd = self.V_prev * np.cos(th)
        yd = self.V_prev * np.sin(th)
        J_inv = np.linalg.inv(np.array([
            [np.cos(th), -self.V_prev * np.sin(th)],
            [np.sin(th), self.V_prev * np.cos(th)]
        ]))
        _acc = np.array([xdd_d, ydd_d])
        _pos = np.array([self.kpx * (x_d - x), self.kpy * (y_d - y)])
        _vel = np.array([self.kpx * (xd_d - xd), self.kpy * (yd_d - yd)])
        a, om = np.matmul(J_inv, _acc + _pos + _vel)
        V = self.V_prev + a * dt
        ########## Code ends here ##########
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        ctrl_out = TurtleBotControl(v=V, omega=om)
        return ctrl_out

    def compute_trajectory_plan(
        self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float
    ) -> TrajectoryPlan | None:
        astar = AStar(
            statespace_lo=(state.x - horizon, state.y - horizon),
            statespace_hi=(state.x + horizon, state.y + horizon),
            x_init=(state.x, state.y), 
            x_goal=(goal.x, goal.y),
            occupancy=occupancy,
            resolution=resolution,
        )
        self.get_logger().info("\n\nsolving\n\n")
        solvable = astar.solve()
        if not solvable:
            self.get_logger().info("\n\nnotsolvable\n\n")
            return None
        if astar.path.shape[0] < 4:
            self.get_logger().info("\n\ntooshort\n\n")
            return None
        self.get_logger().info("\n\nfound\n\n")
        path = compute_smooth_plan(
            path=astar.path,
            k=3,
            v_desired=self.v_desired,
            spline_alpha=self.spline_alpha
        )
        self.reset()
        return path
    



if __name__ == "__main__":
    # yaw = math.pi / 2
    # print(f"convert yaw = {yaw} to quaternion = {yaw_to_quaternion(yaw)}")

    rclpy.init()
    node = MyNavigator()
    rclpy.spin(node)
    rclpy.shutdown()