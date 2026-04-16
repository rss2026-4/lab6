import rclpy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from .utils import LineTrajectory

import numpy as np
import math




class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('drive_topic', "/drive")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0.5  # FILL IN #
        self.speed = 1.0  # FILL IN #
        self.wheelbase_length = 0.325  # FILL IN #
        self.max_steer = 0.34

        self.path_x = None
        self.path_y = None

        self.final_x = None
        self.final_y = None

        self.initialized_traj = False
        self.trajectory = LineTrajectory(self, "/followed_trajectory")
        traj_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 traj_qos)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        # self.actual_pose = self.create_subscription(PoseStamped,
        #                                             '/pf/pose',
        #                                             self.actual_pose_callback,
        #                                             1)
                
        self.get_logger().info(f"Started")
    

    def pose_callback(self, odometry_msg):

        if self.path_x is None or len(self.path_x) < 2:
            return


        
        pose_x = odometry_msg.pose.pose.position.x
        pose_y = odometry_msg.pose.pose.position.y
        pose = np.array([pose_x, pose_y])

        
        self.get_logger().info(f"Pose ({pose})")

        if np.sqrt((pose_x-self.final_x)**2+(pose_y-self.final_y)**2) < 0.5:
            self.get_logger().info(f"I refuse")
            self.publish_drive(0.0,0.0)
            return
        
        min_dist = 1e9
        closest_seg = None

        for i in range(len(self.path_x)-1):
            A = np.array([self.path_x[i], self.path_y[i]])
            B = np.array([self.path_x[i+1], self.path_y[i+1]])
            AB = B - A
            ab_sq = np.dot(AB, AB)
            if ab_sq < 1e-9:
                continue

            t = np.dot(pose - A, AB) / ab_sq
            t = np.clip(t, 0.0,1.0)
            closest = A + t * AB

            dist = np.linalg.norm(pose - closest)
            if dist < min_dist:
                min_dist = dist
                closest_seg = i

        if closest_seg is None:
            self.publish_drive(0.0, 0.0)
            return

        #do the math to get the goal
        Q = np.array([0,0])                  # Centre of circle
        r = self.lookahead                # Radius of circle
        goal_point = None

        for i in range(closest_seg, len(self.path_x)-1):
            A = np.array([self.path_x[i], self.path_y[i]])
            B = np.array([self.path_x[i+1], self.path_y[i+1]])

            P1 = A - pose
            P2 = B - pose
            # self.get_logger().info(f"P1 ({P1})")
            V = P2 - P1
            # self.get_logger().info(f"Vector ({V})")

            a = V.dot(V)
            if a < 1e-9:
                continue
            b = 2 * V.dot(P1-Q)
            c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r**2
            
            #this means the circle does not intersect the line 
            disc = b**2 - 4 * a * c
            if disc < -1e-9:
                # self.get_logger().info(f"returning early")
                # return False, None
                continue
            disc = max(disc, 0.0)

            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            valid_ts = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
            if not valid_ts:
                continue

            t = max(valid_ts)

            # goal_x, goal_y = (P1 + t * V) + pose
            goal_point = A + t * (B - A)
            break

        if goal_point is None:
            goal_point = np.array([self.final_x, self.final_y])

        goal_x, goal_y = goal_point

        self.get_logger().info(f"current location ({pose_x},{pose_y})")
        self.get_logger().info(f"goal location ({goal_x},{goal_y})")

        q = odometry_msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        # Transform goal point from world frame to vehicle frame
        dx = goal_x - pose_x
        dy = goal_y - pose_y

        # Rotate into car frame
        goal_x_car =  np.cos(-yaw) * dx - np.sin(-yaw) * dy
        goal_y_car =  np.sin(-yaw) * dx + np.cos(-yaw) * dy

        self.pure_pursuit(goal_x_car, goal_y_car)


    def publish_drive(self,steering, cur_speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.frame_id = "base_link"
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = steering
        self.get_logger().info(f"steering angle {steering}")
        drive_msg.drive.speed = cur_speed
        self.drive_pub.publish(drive_msg)


    def pure_pursuit(self, path_x, path_y):

        alpha = np.arctan2(path_y, path_x)
        lookahead_dist = np.hypot(path_x, path_y)
        if lookahead_dist < 1e-6:
            self.publish_drive(0.0, 0.0)
            return

        steering_angle = np.arctan2(
            2.0 * self.wheelbase_length * np.sin(alpha),
            lookahead_dist
        )
        steering_angle = float(np.clip(steering_angle, -self.max_steer, self.max_steer))

        self.publish_drive(steering_angle, self.speed)


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        if len(msg.poses) < 2:
            self.path_x = None
            self.path_y = None
            self.final_x = None
            self.final_y = None
            self.initialized_traj = False
            self.publish_drive(0.0, 0.0)
            return

        self.initialized_traj = True

        # processing trajectory into list of points
        self.path_x = [pose.position.x for pose in msg.poses]
        self.path_y = [pose.position.y for pose in msg.poses]
            
        self.final_x = self.path_x[-1]
        self.final_y = self.path_y[-1]


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
