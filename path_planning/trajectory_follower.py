import rclpy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from .utils import LineTrajectory

from visualization_msgs.msg import Marker
import numpy as np
from scipy.interpolate import BSpline




class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        # self.declare_parameter('odom_topic', "default")
        # self.declare_parameter('drive_topic', "default")
        self.declare_parameter('odom_topic', "/odom_topic")
        self.declare_parameter('drive_topic', "/drive_topic")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0  # FILL IN #
        self.speed = 1  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #

        self.initialized_traj = False
        self.trajectory = LineTrajectory(self, "/followed_trajectory")

        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
        self.car_marker_pub = self.create_publisher(Marker,
                                                    '/pure_pursuit/car_marker',
                                                    1)
    

    def pose_callback(self, odometry_msg):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'car'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose = odometry_msg.pose.pose

        # Size of racecar
        marker.scale.x = 0.4  # length
        marker.scale.y = 0.2  # width
        marker.scale.z = 0.1  # height

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.car_marker_pub.publish(marker)

    def points_to_spline(traj):
        '''takes in a trajectory and spits out a spline'''
        # Define knot vector, coefficients, and degree
        t = [0, 1, 2, 3, 4, 5]
        c = [-1, 2, 0, -1]
        k = 2

        # Create a BSpline object
        spl = BSpline(t, c, k)

        # Evaluate the spline at multiple points
        x = np.linspace(1.5, 4.5, 50)
        y = spl(x)



    
    def publish_drive(self,steering):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.frame_id = "base_link"
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = 0
        drive_msg.drive.speed = self.speed
        self.drive_pub.publish(drive_msg)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
