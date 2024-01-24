#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.task import Future

from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.math_utils import wrap_angle


class PlottingNode(Node):
    def __init__(self, des_angle=1.274, recording_duration=5.0):
        super().__init__("plot_node")
        
        self.goal = des_angle

        self.pub = self.create_publisher(TurtleBotState, "/cmd_pose", 10)
        self.pub_timer = self.create_timer(0.25, self.goal_pub_cb)

        self.status = Future()

        self.plot_sub = self.create_subscription(
            TurtleBotState, "/state", self.plot_cb, 10
        )

        self.angles = []
        self.recording_duration = recording_duration

        self.start_time = self.get_clock().now().nanoseconds * 1e-9

    def goal_pub_cb(self):
        msg = TurtleBotState()
        msg.theta = self.goal
        self.pub.publish(msg)


    def plot_cb(self, msg: TurtleBotState):
        """
        Record data until it's been longer than the set time.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.start_time < self.recording_duration:
            self.angles.append(wrap_angle(msg.theta))
        else:
            self.status.set_result(True)

    def plot(self):
        num_msgs = len(self.angles)
        theta_traj = np.array(self.angles)
        time = np.linspace(0, self.recording_duration, num=num_msgs)
        goal_thetas = self.goal * np.ones_like(time)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(time, theta_traj, linewidth=2, label="theta", c="cornflowerblue")
        ax.plot(time, goal_thetas, linewidth=2, label="goal", c="orange")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Theta")
        fig.legend()
        filename = Path("src/p3_output.png")
        try:
            fig.savefig(filename)  # save the figure to file
        except OSError as e:
            print(
                f"Tried to plot to {filename.absolute()}, but directory does not exist!"
            )
            print("Could not save plot, make sure to launch from ~/autonomy_ws!")
        else:
            print(f"Successfully plotted to {filename.absolute()}")
        plt.close(fig)


if __name__ == "__main__":
    rclpy.init()
    node = PlottingNode(recording_duration=10.0)
    rclpy.spin_until_future_complete(node, node.status, timeout_sec=20)
    node.plot()
    rclpy.shutdown()
