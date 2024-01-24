#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

# class HeadingController(Node):
class HeadingController(BaseHeadingController):
    def __init__(self, node_name: str = "heading_controller") -> None:
        super().__init__(node_name)
        # self.Kp = 2.0
        self.declare_parameter("Kp", 2.0)
    
    @property
    def Kp(self) -> float:
        return self.get_parameter("Kp").value

    def compute_control_with_goal(self,
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
    
if __name__ == "__main__":
    rclpy.init()
    heading_control = HeadingController()
    rclpy.spin(heading_control)
    # heading_control.destroy_node()
    rclpy.shutdown()

        