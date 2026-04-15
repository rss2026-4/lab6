
import rclpy
import numpy as np
import time

from geometry_msgs.msg import Point, PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker


class PathPlan(Node):
    """Listens for goal pose published by RViz and uses it to plan a path from
    current car pose using RRT-Connect.
    """

    STEP_SIZE = 0.5                    # max branch length (meters)
    GOAL_THRESHOLD = 0.5               # goal radius (meters)
    MAX_ITERATIONS = 10000
    COLLISION_CHECK_RESOLUTION = 0.05  # sample every this distance along segments (meters)
    SHORTCUT_ITERATIONS = 200
    SHORTCUT_MARGIN = 0.3              # lateral safety margin for shortcut checks (meters)
    VIZ_EVERY_N = 5

    # smoothing / resampling
    MIN_TURN_RADIUS = 0.9              # L / tan(max_steering) ≈ 0.325 / tan(0.34)
    RESAMPLE_SPACING = 0.15            # dense resampling before smoothing (meters)
    SMOOTH_ITERATIONS = 500
    SMOOTH_WEIGHT = 0.3                # how aggressively to pull waypoints toward smooth line

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 1)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.traj_pub = self.create_publisher(PoseArray, "/trajectory/current", 10)
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_cb, 10)
        self.tree_a_pub = self.create_publisher(Marker, "/rrt/tree_a", 1)
        self.tree_b_pub = self.create_publisher(Marker, "/rrt/tree_b", 1)
        self.initialpose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initialpose_cb, 10)

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map_info = None
        self.occupancy_grid = None
        self.free_cells = None
        self.car_pose = None

        self.get_logger().info("RRT-Connect planner initialized")

    # ── Map handling ──────────────────────────────────────────────────────

    def map_cb(self, msg):
        """Fires when the map arrives. Reshape into 2D array and cache free cells."""
        self.map_info = msg.info
        self.occupancy_grid = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        self.free_cells = np.argwhere(self.occupancy_grid == 0)

        o = msg.info.origin
        q = o.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.map_origin_x = o.position.x
        self.map_origin_y = o.position.y
        self.map_cos = np.cos(yaw)
        self.map_sin = np.sin(yaw)

        self.get_logger().info("Map received")

    # ── Coordinate transforms ─────────────────────────────────────────────

    def world_to_grid(self, x, y):
        dx = x - self.map_origin_x
        dy = y - self.map_origin_y
        mx = self.map_cos * dx + self.map_sin * dy
        my = -self.map_sin * dx + self.map_cos * dy
        u = int(np.floor(mx / self.map_info.resolution))
        v = int(np.floor(my / self.map_info.resolution))
        return u, v

    def grid_to_world(self, u, v):
        mx = u * self.map_info.resolution
        my = v * self.map_info.resolution
        x = self.map_cos * mx - self.map_sin * my + self.map_origin_x
        y = self.map_sin * mx + self.map_cos * my + self.map_origin_y
        return x, y

    # ── Collision helpers ─────────────────────────────────────────────────

    def is_free(self, x, y, strict=False):
        """Check if a point is not an obstacle.
        strict=False: allows soft-cost cells (value < 100) for planning.
        strict=True:  only truly free cells (value == 0).
        """
        u, v = self.world_to_grid(x, y)
        if 0 <= u < self.map_info.width and 0 <= v < self.map_info.height:
            val = self.occupancy_grid[v, u]
            if strict:
                return val == 0
            return val < 100
        return False

    def collision_free(self, p1, p2, margin=0.0):
        """Check if line segment p1→p2 is collision free.
        If margin > 0, also checks points offset laterally by that distance
        to prevent wall-grazing shortcuts.
        """
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 1e-6:
            return self.is_free(p1[0], p1[1])
        n_checks = max(2, int(dist / self.COLLISION_CHECK_RESOLUTION))

        dx, dy = (p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist
        nx, ny = -dy, dx  # unit normal to the segment

        for i in range(n_checks + 1):
            t = i / n_checks
            cx = p1[0] + t * (p2[0] - p1[0])
            cy = p1[1] + t * (p2[1] - p1[1])
            if not self.is_free(cx, cy):
                return False
            if margin > 0:
                if not self.is_free(cx + nx * margin, cy + ny * margin):
                    return False
                if not self.is_free(cx - nx * margin, cy - ny * margin):
                    return False
        return True

    # ── Pose callbacks ────────────────────────────────────────────────────

    def pose_cb(self, msg):
        self.car_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def initialpose_cb(self, msg):
        self.clear_trees()
        self.trajectory.clear()
        self.trajectory.publish_viz()
        self.get_logger().info("Initial pose set, cleared previous plan")

    # ── Goal callback ─────────────────────────────────────────────────────

    def goal_cb(self, msg):
        if self.occupancy_grid is None:
            self.get_logger().warn("No map received yet, cannot plan")
            return
        if self.car_pose is None:
            self.get_logger().warn("No car pose received yet, cannot plan")
            return
        goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f"Planning from ({self.car_pose[0]:.2f}, {self.car_pose[1]:.2f}) "
            f"to ({goal[0]:.2f}, {goal[1]:.2f})"
        )
        self.plan_path(self.car_pose, goal)

    # ── RRT-Connect core ─────────────────────────────────────────────────

    def nearest(self, tree_points, point):
        """Return index of nearest node in tree to point."""
        diffs = tree_points[:len(tree_points)] - np.array(point)
        dists = np.einsum('ij,ij->i', diffs, diffs)
        return int(np.argmin(dists))

    def steer(self, from_pt, to_pt):
        """Step from from_pt toward to_pt by at most STEP_SIZE."""
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        dist = np.hypot(dx, dy)
        if dist <= self.STEP_SIZE:
            return to_pt
        ratio = self.STEP_SIZE / dist
        return (from_pt[0] + dx * ratio, from_pt[1] + dy * ratio)

    def extend(self, tree_points, tree_parents, point):
        """Extend tree toward point. Returns (status, new_node_index)."""
        near_idx = self.nearest(tree_points, point)
        near_pt = (tree_points[near_idx][0], tree_points[near_idx][1])
        new_pt = self.steer(near_pt, point)

        if self.collision_free(near_pt, new_pt):
            new_idx = len(tree_points)
            tree_points.append(np.array(new_pt))
            tree_parents.append(near_idx)
            if np.hypot(new_pt[0] - point[0], new_pt[1] - point[1]) < 1e-6:
                return "reached", new_idx
            return "advanced", new_idx
        return "trapped", -1

    def connect(self, tree_points, tree_parents, point):
        """Greedily extend tree toward point until reached or trapped."""
        while True:
            status, idx = self.extend(tree_points, tree_parents, point)
            if status != "advanced":
                return status, idx

    def extract_path(self, tree_a_points, tree_a_parents,
                     tree_b_points, tree_b_parents, a_idx, b_idx):
        """Extract full path from start to goal by merging both trees."""
        path_a = []
        idx = a_idx
        while idx != -1:
            path_a.append((tree_a_points[idx][0], tree_a_points[idx][1]))
            idx = tree_a_parents[idx]
        path_a.reverse()

        path_b = []
        idx = b_idx
        while idx != -1:
            path_b.append((tree_b_points[idx][0], tree_b_points[idx][1]))
            idx = tree_b_parents[idx]

        return path_a + path_b

    def sample_free(self):
        """Sample a random free grid cell and convert to world coordinates."""
        if self.free_cells is None or len(self.free_cells) == 0:
            return None
        idx = np.random.randint(len(self.free_cells))
        v, u = self.free_cells[idx]
        return self.grid_to_world(u + 0.5, v + 0.5)

    # ── Path post-processing ─────────────────────────────────────────────

    def shortcut_path(self, path):
        """Iteratively shortcut the path. Uses lateral margin to prevent
        shortcuts that graze walls.
        """
        if len(path) <= 2:
            return path
        path = list(path)
        for _ in range(self.SHORTCUT_ITERATIONS):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            if self.collision_free(path[i], path[j], margin=self.SHORTCUT_MARGIN):
                path = path[:i + 1] + path[j:]
        return path

    def resample_path(self, path):
        """Resample path into evenly spaced points so smoothing works uniformly."""
        if len(path) < 2:
            return path
        resampled = [path[0]]
        leftover = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            seg_len = np.hypot(dx, dy)
            if seg_len < 1e-9:
                continue
            ux, uy = dx / seg_len, dy / seg_len
            consumed = self.RESAMPLE_SPACING - leftover
            while consumed <= seg_len:
                resampled.append((
                    path[i - 1][0] + ux * consumed,
                    path[i - 1][1] + uy * consumed,
                ))
                consumed += self.RESAMPLE_SPACING
            leftover = seg_len - (consumed - self.RESAMPLE_SPACING)
        resampled.append(path[-1])
        return resampled

    @staticmethod
    def turning_radius(p0, p1, p2):
        """Radius of the circle through three consecutive points (Menger curvature)."""
        ax, ay = p1[0] - p0[0], p1[1] - p0[1]
        bx, by = p2[0] - p1[0], p2[1] - p1[1]
        cross = abs(ax * by - ay * bx)
        if cross < 1e-12:
            return float("inf")
        a = np.hypot(ax, ay)
        b = np.hypot(bx, by)
        c = np.hypot(p2[0] - p0[0], p2[1] - p0[1])
        return (a * b * c) / (2.0 * cross)

    def smooth_path(self, path):
        """Iteratively smooth while enforcing minimum turning radius
        and collision safety. Falls back to original if smoothing
        introduces a collision.
        """
        if len(path) <= 2:
            return path
        pts = [list(p) for p in path]
        for _ in range(self.SMOOTH_ITERATIONS):
            for i in range(1, len(pts) - 1):
                new_x = pts[i][0] + self.SMOOTH_WEIGHT * (
                    pts[i - 1][0] + pts[i + 1][0] - 2.0 * pts[i][0])
                new_y = pts[i][1] + self.SMOOTH_WEIGHT * (
                    pts[i - 1][1] + pts[i + 1][1] - 2.0 * pts[i][1])
                r = self.turning_radius(pts[i - 1], (new_x, new_y), pts[i + 1])
                if r >= self.MIN_TURN_RADIUS:
                    pts[i][0] = new_x
                    pts[i][1] = new_y
        # verify collision-free; fall back to original if not
        for i in range(len(pts) - 1):
            if not self.collision_free(pts[i], pts[i + 1]):
                return path
        return [tuple(p) for p in pts]

    # ── Visualization ─────────────────────────────────────────────────────

    def publish_tree(self, publisher, tree_points, tree_parents, r, g, b, marker_id):
        marker = Marker()
        marker.header = Header(frame_id="/map", stamp=self.get_clock().now().to_msg())
        marker.ns = "rrt_trees"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.color = ColorRGBA(r=r, g=g, b=b, a=0.6)
        for i in range(1, len(tree_points)):
            parent = tree_parents[i]
            p1 = Point(x=tree_points[parent][0], y=tree_points[parent][1], z=0.0)
            p2 = Point(x=tree_points[i][0], y=tree_points[i][1], z=0.0)
            marker.points.append(p1)
            marker.points.append(p2)
        publisher.publish(marker)

    def clear_trees(self):
        for pub, mid in [(self.tree_a_pub, 0), (self.tree_b_pub, 1)]:
            marker = Marker()
            marker.header = Header(frame_id="/map", stamp=self.get_clock().now().to_msg())
            marker.ns = "rrt_trees"
            marker.id = mid
            marker.action = Marker.DELETE
            pub.publish(marker)

    def clear_current_plan(self):
        """Clear the currently published plan so the follower does not use stale data."""
        self.trajectory.clear()
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()
        self.clear_trees()

    # ── Main planning loop ────────────────────────────────────────────────

    def plan_path(self, start_point, end_point):
        self.clear_trees()
        self.trajectory.clear()
        self.trajectory.publish_viz()

        if not self.is_free(start_point[0], start_point[1]):
            self.get_logger().error("Start point is in an obstacle!")
            return
        if not self.is_free(end_point[0], end_point[1]):
            self.get_logger().error("Goal point is in an obstacle!")
            return

        t_start = time.time()

        tree_a_points = [np.array(start_point)]
        tree_a_parents = [-1]
        tree_b_points = [np.array(end_point)]
        tree_b_parents = [-1]

        swapped = False

        start_tree = tree_a_points
        start_parents = tree_a_parents
        goal_tree = tree_b_points
        goal_parents = tree_b_parents

        for i in range(self.MAX_ITERATIONS):
            q_rand = self.sample_free()
            if q_rand is None:
                continue

            status_a, idx_a = self.extend(tree_a_points, tree_a_parents, q_rand)
            if status_a != "trapped":
                new_pt = (tree_a_points[idx_a][0], tree_a_points[idx_a][1])
                status_b, idx_b = self.connect(tree_b_points, tree_b_parents, new_pt)

                if status_b == "reached":
                    self.publish_tree(self.tree_a_pub, start_tree, start_parents, 1.0, 0.0, 0.0, 0)
                    self.publish_tree(self.tree_b_pub, goal_tree, goal_parents, 1.0, 0.0, 0.0, 1)

                    elapsed = time.time() - t_start
                    self.get_logger().info(
                        f"RRT-Connect found path in {i+1} iterations, {elapsed:.2f}s")

                    if swapped:
                        raw_path = self.extract_path(
                            tree_b_points, tree_b_parents,
                            tree_a_points, tree_a_parents, idx_b, idx_a)
                    else:
                        raw_path = self.extract_path(
                            tree_a_points, tree_a_parents,
                            tree_b_points, tree_b_parents, idx_a, idx_b)

                    path = self.shortcut_path(raw_path)
                    path = self.resample_path(path)
                    path = self.smooth_path(path)
                    self.get_logger().info(
                        f"Path: {len(raw_path)} raw -> {len(path)} after shortcut/resample/smooth")

                    self.trajectory.clear()
                    for pt in path:
                        self.trajectory.addPoint(pt)

                    self.traj_pub.publish(self.trajectory.toPoseArray())
                    self.trajectory.publish_viz()
                    return

            if i % self.VIZ_EVERY_N == 0:
                self.publish_tree(self.tree_a_pub, start_tree, start_parents, 1.0, 0.0, 0.0, 0)
                self.publish_tree(self.tree_b_pub, goal_tree, goal_parents, 1.0, 0.0, 0.0, 1)

            tree_a_points, tree_b_points = tree_b_points, tree_a_points
            tree_a_parents, tree_b_parents = tree_b_parents, tree_a_parents
            swapped = not swapped

        elapsed = time.time() - t_start
        self.get_logger().error(
            f"RRT-Connect failed after {self.MAX_ITERATIONS} iterations ({elapsed:.2f}s)")
        self.clear_current_plan()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()

# import rclpy

# from geometry_msgs.msg import PoseArray, PoseStamped
# from nav_msgs.msg import OccupancyGrid, Odometry
# from path_planning.utils import LineTrajectory
# from rclpy.node import Node


# class PathPlan(Node):
#     """ Listens for goal pose published by RViz and uses it to plan a path from
#     current car pose.
#     """

#     def __init__(self):
#         super().__init__("trajectory_planner")
#         self.declare_parameter('odom_topic', "default")
#         self.declare_parameter('map_topic', "default")

#         self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
#         self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value

#         self.map_sub = self.create_subscription(
#             OccupancyGrid,
#             self.map_topic,
#             self.map_cb,
#             1)

#         self.goal_sub = self.create_subscription(
#             PoseStamped,
#             "/goal_pose",
#             self.goal_cb,
#             10
#         )

#         self.traj_pub = self.create_publisher(
#             PoseArray,
#             "/trajectory/current",
#             10
#         )

#         self.pose_sub = self.create_subscription(
#             Odometry,
#             self.odom_topic,
#             self.pose_cb,
#             10
#         )

#         self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

#     def map_cb(self, msg):
#         raise NotImplementedError

#     def pose_cb(self, pose):
#         raise NotImplementedError

#     def goal_cb(self, msg):
#         raise NotImplementedError

#     def plan_path(self, start_point, end_point, map):
#         self.traj_pub.publish(self.trajectory.toPoseArray())
#         self.trajectory.publish_viz()


# def main(args=None):
#     rclpy.init(args=args)
#     planner = PathPlan()
#     rclpy.spin(planner)
#     rclpy.shutdown()
