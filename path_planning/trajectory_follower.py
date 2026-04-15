import rclpy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from .utils import LineTrajectory

from visualization_msgs.msg import Marker
import numpy as np
from scipy.interpolate import splprep, splev
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
        closest_seg = 0

        for i in range(len(self.path_x)-1):
            A = np.array([self.path_x[i], self.path_y[i]])
            B = np.array([self.path_x[i+1], self.path_y[i+1]])
            AB = B - A
            t = np.dot(pose - A, AB) / np.dot(AB, AB)
            t = np.clip(t, 0.0,1.0)
            closest = A + t * AB

            dist = np.linalg.norm(pose - closest)
            if dist < min_dist:
                min_dist = dist
                closest_seg = i


        #do the math to get the goal
        Q = np.array([0,0])                  # Centre of circle
        r = self.lookahead                # Radius of circle

        for i in range(closest_seg, len(self.path_x)-1):
            A = np.array([self.path_x[i], self.path_y[i]])
            B = np.array([self.path_x[i+1], self.path_y[i+1]])

            P1 = A - pose
            P2 = B - pose
            # self.get_logger().info(f"P1 ({P1})")
            V = P2 - P1
            # self.get_logger().info(f"Vector ({V})")

            a = V.dot(V)
            b = 2 * V.dot(P1-Q)
            c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r**2
            
            #this means the circle does not intersect the line 
            disc = b**2 - 4 * a * c
            if disc < 0:
                # self.get_logger().info(f"returning early")
                # return False, None
                continue

            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            valid_ts = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
            if not valid_ts:
                continue

            t = max(valid_ts)

            # goal_x, goal_y = (P1 + t * V) + pose
            goal_x, goal_y = A + t * (B - A)

            self.get_logger().info(f"current location ({pose_x},{pose_y})")
            self.get_logger().info(f"goal location ({goal_x},{goal_y})")

            q = odometry_msg.pose.pose.orientation
            yaw = 2 * np.arctan2(q.z, q.w)   # works for planar robots
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

        steering_angle = np.arctan2(
            2.0 * self.wheelbase_length * np.sin(alpha),
            self.lookahead
        )

        self.publish_drive(steering_angle, self.speed)


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.initialized_traj = True

        # processing trajectory into list of points
        self.path_x = []
        self.path_y = []
        for i, pose in enumerate(msg.poses):
            self.path_x.append(pose.position.x)
            self.path_y.append(pose.position.y)
            
        self.final_x = self.path_x[-1]
        self.final_y = self.path_y[-1]


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()






# #!/usr/bin/env python
# import math
# import numpy as np

# import rclpy
# from rclpy.node import Node

# from ackermann_msgs.msg import AckermannDriveStamped
# from geometry_msgs.msg import Pose, PoseArray, Quaternion
# from nav_msgs.msg import Odometry

# from .utils import LineTrajectory  # Adjust the import according to your package structure

# # --- Module-level constants ---
# FINISH_RADIUS = 0.5           # meters; stop when within this radius of the final waypoint
# DECEL_RADIUS = 3.0            # meters; start decelerating when this close to the end
# LOOKAHEAD_MIN = 1.0           # meters; minimum lookahead distance (tight turns)
# LOOKAHEAD_MAX = 1.0           # meters; maximum lookahead distance (straights)
# CAR_LENGTH = 0.325            # meters; vehicle wheelbase length
# MAX_SPEED = 1.5               # m/s; maximum driving speed
# MIN_SPEED = 1.0               # m/s; minimum driving speed (for tight turns)
# MAX_STEERING_ANGLE = 0.34     # radians; max steering angle (~20 degrees)
# CURVATURE_WINDOW = 3.0        # meters; how far ahead to scan for upcoming curvature
# CROSS_TRACK_WARN = 1.5        # meters; slow down if cross-track error exceeds this


# def quaternion_to_yaw(q: Quaternion):
#     """Convert quaternion to yaw angle."""
#     siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
#     cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
#     return math.atan2(siny_cosp, cosy_cosp)


# class PurePursuit(Node):
#     """
#     Pure Pursuit trajectory tracking node.

#     Key features:
#     - Segment-based nearest point (vectorized numpy)
#     - Circle-line-segment intersection for the true lookahead point
#     - Path progress tracking to prevent backtracking on looping trajectories
#     - Adaptive lookahead distance based on upcoming curvature
#     - Curvature look-ahead for proactive speed reduction before sharp turns
#     - Graceful deceleration near the goal
#     - Cross-track error awareness (slows down when far from the path)
#     """

#     def __init__(self):
#         super().__init__("trajectory_follower")

#         # Declare parameters (with defaults)
#         self.declare_parameter('odom_topic', '/odom')
#         self.declare_parameter('drive_topic', '/drive')

#         self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
#         self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

#         # Tunable parameters
#         self.lookahead_min = LOOKAHEAD_MIN
#         self.lookahead_max = LOOKAHEAD_MAX
#         self.max_speed = MAX_SPEED
#         self.min_speed = MIN_SPEED
#         self.wheelbase_length = CAR_LENGTH

#         self.initialized_traj = False
#         self.stopped = True

#         # Trajectory data (populated as numpy arrays)
#         self.traj_points = None       # (N, 2) array of trajectory waypoints
#         self.traj_segments = None     # (N-1, 2) array of segment vectors
#         self.traj_seg_lengths = None  # (N-1,) array of segment lengths
#         self.cum_distances = None     # (N,) cumulative arc-length along trajectory
#         self.total_length = 0.0       # total trajectory length

#         # Path progress tracking — prevents backtracking on looping paths
#         self.last_seg_idx = 0

#         # Create an instance of the trajectory helper (for visualization).
#         self.trajectory = LineTrajectory(self, "/followed_trajectory")

#         # Subscribe to trajectory updates.
#         self.traj_sub = self.create_subscription(
#             PoseArray,
#             "/trajectory/current",
#             self.trajectory_callback,
#             1)

#         # Subscribe to odometry updates.
#         self.odom_sub = self.create_subscription(
#             Odometry,
#             self.odom_topic,
#             self.pose_callback,
#             1)

#         # Create a publisher for drive commands.
#         self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)

#         self.get_logger().info("Pure Pursuit node initialized")

#     # ------------------------------------------------------------------ #
#     #  Trajectory handling
#     # ------------------------------------------------------------------ #
#     def trajectory_callback(self, msg: PoseArray):
#         self.get_logger().info(f"Receiving new trajectory with {len(msg.poses)} points")

#         # Reset and update the visualization helper.
#         self.trajectory.clear()
#         self.trajectory.fromPoseArray(msg)
#         self.trajectory.publish_viz(duration=0.0)

#         # Pre-compute numpy arrays for fast lookup.
#         points = np.array([[p.position.x, p.position.y] for p in msg.poses])
#         if len(points) < 2:
#             self.get_logger().warn("Trajectory has fewer than 2 points; ignoring.")
#             return

#         self.traj_points = points                              # (N, 2)
#         self.traj_segments = np.diff(points, axis=0)           # (N-1, 2)
#         self.traj_seg_lengths = np.linalg.norm(self.traj_segments, axis=1)  # (N-1,)

#         # Cumulative arc-length at each waypoint.
#         self.cum_distances = np.zeros(len(points))
#         self.cum_distances[1:] = np.cumsum(self.traj_seg_lengths)
#         self.total_length = self.cum_distances[-1]

#         # Reset progress tracker for the new trajectory.
#         self.last_seg_idx = 0

#         self.initialized_traj = True
#         self.stopped = False

#     # ------------------------------------------------------------------ #
#     #  Core control loop
#     # ------------------------------------------------------------------ #
#     def pose_callback(self, odom_msg: Odometry):
#         if not self.initialized_traj or self.stopped:
#             return

#         # Vehicle state
#         pos = odom_msg.pose.pose.position
#         yaw = quaternion_to_yaw(odom_msg.pose.pose.orientation)
#         car = np.array([pos.x, pos.y])

#         # --- Check if close to end of trajectory ---
#         dist_to_end = np.linalg.norm(self.traj_points[-1] - car)
#         if dist_to_end < FINISH_RADIUS:
#             self.publish_drive_command(0.0, 0.0)
#             self.stopped = True
#             self.get_logger().info("Reached goal. Stopping.")
#             return

#         # Step 1: Find the nearest point on the trajectory (forward-biased).
#         nearest_seg_idx, nearest_point, cross_track_error = \
#             self.find_nearest_point_on_trajectory(car)

#         # Step 2: Compute adaptive lookahead distance.
#         #   - Increase lookahead on straights for smoother tracking
#         #   - Decrease on curves for tighter following
#         #   - Increase when far from path to "recover" back onto it
#         upcoming_curvature = self.estimate_upcoming_curvature(nearest_seg_idx)
#         # Map curvature [0, high] -> lookahead [max, min]
#         curvature_factor = min(upcoming_curvature * self.wheelbase_length, 1.0)  # 0..1
#         lookahead = self.lookahead_max - (self.lookahead_max - self.lookahead_min) * curvature_factor

#         # If far from the path, boost lookahead to help the car rejoin smoothly.
#         if cross_track_error > CROSS_TRACK_WARN:
#             lookahead = max(lookahead, cross_track_error * 1.2)

#         # Step 3: Find the lookahead point via circle-line-segment intersection.
#         lookahead_point = self.find_lookahead_point(car, nearest_seg_idx, lookahead)

#         if lookahead_point is None:
#             self.get_logger().warn("No valid lookahead point found. Stopping.")
#             self.publish_drive_command(0.0, 0.0)
#             return

#         # Step 4: Transform lookahead point into the vehicle frame.
#         dx = lookahead_point[0] - car[0]
#         dy = lookahead_point[1] - car[1]
#         cos_yaw = math.cos(yaw)
#         sin_yaw = math.sin(yaw)
#         local_x =  cos_yaw * dx + sin_yaw * dy
#         local_y = -sin_yaw * dx + cos_yaw * dy

#         # Step 5: Compute steering angle (pure pursuit formula).
#         #   curvature = 2 * local_y / L_d^2
#         #   steering  = atan(curvature * wheelbase)
#         L_d = math.hypot(local_x, local_y)
#         if L_d < 1e-6:
#             self.publish_drive_command(self.max_speed, 0.0)
#             return

#         curvature = 2.0 * local_y / (L_d * L_d)
#         steering_angle = math.atan(self.wheelbase_length * curvature)

#         # Clamp steering angle
#         steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

#         # Step 6: Speed planning.
#         speed = self.compute_speed(steering_angle, dist_to_end, cross_track_error,
#                                    upcoming_curvature)

#         self.publish_drive_command(speed, steering_angle)

#     # ------------------------------------------------------------------ #
#     #  Speed planning
#     # ------------------------------------------------------------------ #
#     def compute_speed(self, steering_angle: float, dist_to_end: float,
#                       cross_track_error: float, upcoming_curvature: float) -> float:
#         """
#         Determine driving speed based on multiple factors:
#           1. Current steering angle magnitude
#           2. Upcoming path curvature (proactive braking)
#           3. Distance to goal (graceful deceleration)
#           4. Cross-track error (slow down when far off-path)
#         """
#         # --- Factor 1: Reactive — current steering demand ---
#         steer_ratio = abs(steering_angle) / MAX_STEERING_ANGLE  # 0..1
#         speed_from_steer = self.max_speed - (self.max_speed - self.min_speed) * steer_ratio

#         # --- Factor 2: Proactive — upcoming curvature ---
#         curv_ratio = min(upcoming_curvature * self.wheelbase_length, 1.0)
#         speed_from_curvature = self.max_speed - (self.max_speed - self.min_speed) * curv_ratio

#         # Take the more conservative of reactive and proactive.
#         speed = min(speed_from_steer, speed_from_curvature)

#         # --- Factor 3: Decelerate near the goal ---
#         if dist_to_end < DECEL_RADIUS:
#             # Linear ramp from current speed down to min_speed at the finish.
#             decel_factor = dist_to_end / DECEL_RADIUS  # 0..1
#             goal_speed = self.min_speed + (speed - self.min_speed) * decel_factor
#             speed = min(speed, goal_speed)

#         # --- Factor 4: Slow down when far from the path ---
#         if cross_track_error > CROSS_TRACK_WARN:
#             recovery_factor = CROSS_TRACK_WARN / cross_track_error  # < 1
#             speed *= recovery_factor

#         return max(speed, self.min_speed)

#     # ------------------------------------------------------------------ #
#     #  Nearest point on trajectory (vectorized, forward-biased)
#     # ------------------------------------------------------------------ #
#     def find_nearest_point_on_trajectory(self, car: np.ndarray):
#         """
#         Find the closest point on the piecewise-linear trajectory to `car`.

#         To prevent backtracking on looping trajectories, the search window is
#         biased forward from the last known segment index: we search from
#         a few segments behind `last_seg_idx` all the way to the end.

#         Returns (segment_index, closest_point, cross_track_error).
#         """
#         n_segs = len(self.traj_seg_lengths)

#         # Allow a small backward window to handle noise / slight reversals.
#         BACK_WINDOW = 5
#         search_start = max(0, self.last_seg_idx - BACK_WINDOW)

#         # Slice the arrays for the search window.
#         pts = self.traj_points[search_start:]        # includes extra endpoint
#         segs = self.traj_segments[search_start:]
#         seg_lens = self.traj_seg_lengths[search_start:]

#         # Vector from each segment start to the car.
#         to_car = car - pts[:-1]                                          # (M, 2)

#         # Project onto each segment: t = dot(to_car, seg) / |seg|^2
#         seg_len_sq = seg_lens ** 2
#         safe_len_sq = np.where(seg_len_sq < 1e-12, 1.0, seg_len_sq)
#         t = np.sum(to_car * segs, axis=1) / safe_len_sq
#         t = np.clip(t, 0.0, 1.0)

#         # Closest point on each segment.
#         proj = pts[:-1] + t[:, np.newaxis] * segs

#         # Distances from car to each projected point.
#         dists = np.linalg.norm(proj - car, axis=1)

#         best_local = int(np.argmin(dists))
#         best_global = search_start + best_local
#         cross_track_error = float(dists[best_local])

#         # Update progress tracker.
#         self.last_seg_idx = best_global

#         return best_global, proj[best_local], cross_track_error

#     # ------------------------------------------------------------------ #
#     #  Upcoming curvature estimation
#     # ------------------------------------------------------------------ #
#     def estimate_upcoming_curvature(self, seg_idx: int) -> float:
#         """
#         Estimate the maximum curvature in the next CURVATURE_WINDOW meters
#         along the trajectory from `seg_idx`.

#         Curvature at each waypoint is approximated from the angle change
#         between consecutive segments divided by the average segment length.
#         """
#         n_segs = len(self.traj_seg_lengths)
#         if seg_idx >= n_segs - 1:
#             return 0.0

#         max_curvature = 0.0
#         dist_scanned = 0.0

#         for i in range(seg_idx, n_segs - 1):
#             dist_scanned += self.traj_seg_lengths[i]
#             if dist_scanned > CURVATURE_WINDOW:
#                 break

#             # Angle between segment i and segment i+1.
#             len_a = self.traj_seg_lengths[i]
#             len_b = self.traj_seg_lengths[i + 1]
#             if len_a < 1e-6 or len_b < 1e-6:
#                 continue

#             # Use cross product and dot product for the turning angle.
#             a = self.traj_segments[i]
#             b = self.traj_segments[i + 1]
#             cross = a[0] * b[1] - a[1] * b[0]
#             dot = a[0] * b[0] + a[1] * b[1]
#             angle = abs(math.atan2(cross, dot))

#             # Approximate curvature = angle / average_segment_length
#             avg_len = 0.5 * (len_a + len_b)
#             curvature = angle / avg_len
#             max_curvature = max(max_curvature, curvature)

#         return max_curvature

#     # ------------------------------------------------------------------ #
#     #  Lookahead point via circle-segment intersection
#     # ------------------------------------------------------------------ #
#     def find_lookahead_point(self, car: np.ndarray, start_seg: int, lookahead: float):
#         """
#         Starting from `start_seg`, search forward along the trajectory for the
#         first intersection between the lookahead circle and a trajectory segment.
#         If a segment has two intersections, pick the one farther along the path.

#         Falls back to the last trajectory point if nothing is found.
#         """
#         n_segs = len(self.traj_seg_lengths)

#         for i in range(start_seg, n_segs):
#             p1 = self.traj_points[i]
#             p2 = self.traj_points[i + 1]
#             result = self._circle_line_intersection(car, lookahead, p1, p2)
#             if result is not None:
#                 return result

#         # Fallback: return the last trajectory point.
#         return self.traj_points[-1].copy()

#     @staticmethod
#     def _circle_line_intersection(center, radius, p1, p2):
#         """
#         Find the intersection(s) between a circle (center, radius) and the
#         line segment from p1 to p2.  Returns the point with the largest t
#         in [0, 1] (farthest along the segment), or None if no intersection.
#         """
#         d = p2 - p1
#         f = p1 - center

#         a = np.dot(d, d)
#         b = 2.0 * np.dot(f, d)
#         c = np.dot(f, f) - radius * radius

#         if a < 1e-12:
#             return None  # degenerate segment

#         discriminant = b * b - 4.0 * a * c
#         if discriminant < 0:
#             return None

#         sqrt_disc = math.sqrt(discriminant)
#         t1 = (-b - sqrt_disc) / (2.0 * a)
#         t2 = (-b + sqrt_disc) / (2.0 * a)

#         # Prefer the largest valid t (farthest forward on the path).
#         for t in [t2, t1]:
#             if 0.0 <= t <= 1.0:
#                 return p1 + t * d

#         return None

#     # ------------------------------------------------------------------ #
#     #  Drive command helper
#     # ------------------------------------------------------------------ #
#     def publish_drive_command(self, speed: float, steering_angle: float):
#         cmd = AckermannDriveStamped()
#         cmd.header.stamp = self.get_clock().now().to_msg()
#         cmd.header.frame_id = "base_link"
#         cmd.drive.speed = float(speed)
#         cmd.drive.steering_angle = float(steering_angle)
#         self.drive_pub.publish(cmd)


# def main(args=None):
#     rclpy.init(args=args)
#     node = PurePursuit()
#     rclpy.spin(node)
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()