#!/usr/bin/env python3
from typing import Optional

import numpy as np
from scipy.signal import convolve2d

import rclpy  # ROS Client Library for Python
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool

from nav_msgs.msg import OccupancyGrid

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer_node")  # Initialize the object using the parent class's initialization logic
        self.get_logger().info("Marker publisher has been created!")

        self.state: Optional[TurtleBotState] = None
        self.occupancy: Optional[StochOccupancyGrid2D] = None
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)

        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)

        # self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.handle_nav_success, 10)
        self.timer = self.create_timer(10, self.publish_new_goal)

    def state_callback(self, msg: TurtleBotState) -> None:
        """ callback triggered when receiving latest turtlebot state

        Args:
            msg (TurtleBotState): latest turtlebot state
        """
        self.state = msg

    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

        if self.occupancy is None:
            # publish the initial frontier state
            next_state = self.explore(occupancy)
            self.cmd_nav_pub.publish(next_state)

        self.occupancy = occupancy

    def handle_nav_success(self, msg_in: Bool):
        if not msg_in.data:
            # KILL msg is false
            return

        # KILL msg is true
        # TODO
        next_state = self.explore(self.occupancy)
        self.cmd_nav_pub.publish(next_state)

    def publish_new_goal(self):
        if self.occupancy is not None:
            next_state = self.explore(self.occupancy)
            self.cmd_nav_pub.publish(next_state)
            self.next_state = next_state

    def explore(self, occupancy):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.

        HINTS:
        - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
        - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
        """

        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
        ########################### Code starts here ###########################
        center_idx = (window_size - 1) // 2
        window = np.ones((window_size, window_size))
        window[center_idx, center_idx] = 0

        out = convolve2d(occupancy.probs.T == -1, window, mode="same")
        unknown_ok = out > ((window_size * window_size) - 1) * 0.2

        out = convolve2d(occupancy.probs.T > 0, window, mode="same")
        occupied_ok = out == 0

        out = convolve2d(occupancy.probs.T == 0, window, mode="same")
        unoccupied_ok = out > ((window_size * window_size) - 1) * 0.3

        frontier = np.argwhere(unknown_ok & occupied_ok & unoccupied_ok)
        frontier_states = occupancy.grid2state(frontier)

        dists = np.linalg.norm(frontier_states - np.array([self.state.x, self.state.y]), axis=1)
        print(frontier_states, self.state)

        # if len(dists) == 0:
        #     self.get_logger().info("STOPPING!!")
        #     self.timer.cancel()
        #     ret = TurtleBotState()
        #     ret.x = 0.0
        #     ret.y = 0.0
        #     return ret
        self.get_logger().info("Number of explorable spots: " + str(frontier.size))
        if frontier.size == 0:
            self.get_logger().info("Frontier is empty!")
            self.get_logger().info("STOPPING!!")
            self.timer.cancel()
            return self.next_state
        

        print("Distance to closest frontier state:", min(dists))
        ########################### Code ends here ###########################
        ret = TurtleBotState()
        ret.x = frontier_states[0, 0]
        ret.y = frontier_states[0, 1]
        return ret


def main(args=None):
    rclpy.init(args=args)
    publisher = FrontierExplorer()
    rclpy.spin(publisher)  # Puts the node in an infinite spin loop
    rclpy.shutdown()


if __name__ == "__main__":
    main()
