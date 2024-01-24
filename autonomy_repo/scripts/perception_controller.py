#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool

# class HeadingController(Node):
class PerceptionController(BaseHeadingController):
    def __init__(self, node_name: str = "perception_controller") -> None:
        super().__init__(node_name)
        # self.Kp = 2.0
        self.declare_parameter("Kp", 2.0)
        self.declare_parameter("activate", True)
        self.image_detected = False #("image_detected", False)
        self.create_subscription(Bool, "/detector_bool", self.image_detected_callback, 10)
    
    @property
    def Kp(self) -> float:
        return self.get_parameter("Kp").value
    
    @property
    def activate(self) -> bool:
        return self.get_parameter("activate").value

    # @property
    # def image_detected(self) -> bool:
    #     msg = Bool()
    #     msg.data = self.
    #     return self.get_parameter("image_detected").value
    def image_detected_callback(self, msg: Bool) -> None:
        found = msg.data
        if found:
            self.image_detected = True
        else:
            self.image_detected = False

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
        
        # control_command = TurtleBotControl()
        # th_state = state.theta
        # th_goal = goal.theta
        # th_err = wrap_angle(th_goal - th_state)
        # ang_velo = self.Kp*th_err
        # control_command.omega = ang_velo

        control_command = TurtleBotControl()
        ang_vel = 0.2
        if self.image_detected:
            ang_vel = 0.0
        else:
            ang_vel = 0.2

        control_command.omega = ang_vel
        return control_command
    
if __name__ == "__main__":
    rclpy.init()
    perception_controller = PerceptionController()
    rclpy.spin(perception_controller)
    # heading_control.destroy_node()
    rclpy.shutdown()

        