import rclpy
import numpy as np
import cv2

from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile


class MapInflator(Node):
    """Subscribes to /map, inflates occupied pixels, and republishes as /inflated_map."""

    INFLATION_RADIUS_PX = 12

    def __init__(self):
        super().__init__("map_inflator")
        self.declare_parameter('inflation_radius', self.INFLATION_RADIUS_PX)
        self.inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().integer_value

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_cb, latched_qos)
        self.map_pub = self.create_publisher(OccupancyGrid, "/inflated_map", latched_qos)

        self.get_logger().info(f"Map inflator ready, inflation_radius={self.inflation_radius}px")

    def map_cb(self, msg):
        raw = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        occupied = np.zeros_like(raw, dtype=np.uint8)
        occupied[raw == 100] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.inflation_radius + 1, 2 * self.inflation_radius + 1))
        inflated = cv2.dilate(occupied, kernel)

        out = raw.copy()
        out[inflated == 255] = 100

        out_msg = OccupancyGrid()
        out_msg.header = msg.header
        out_msg.info = msg.info
        out_msg.data = out.flatten().tolist()
        self.map_pub.publish(out_msg)

        self.get_logger().info(
            f"Published inflated map: {msg.info.width}x{msg.info.height}, "
            f"inflation={self.inflation_radius}px "
            f"({self.inflation_radius * msg.info.resolution:.2f}m)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MapInflator()
    rclpy.spin(node)
    rclpy.shutdown()
