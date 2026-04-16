import rclpy
import numpy as np
import time

from geometry_msgs.msg import Point, PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker


class PathPlan(Node):
    """plans from the car pose to the rviz goal with rrt-connect."""

    STEP_SIZE = 0.5 # each new branch of tree is at most this distance long (meters)
    GOAL_THRESHOLD = 0.5 # this is the radius of the goal (meters)
    MAX_ITERATIONS = 10000 # self explanatory
    COLLISION_CHECK_RESOLUTION = 0.1 # when checking if line segment is collision free, sample point every this distance (meters)
    SHORTCUT_ITERATIONS = 200 # self explanatory
    VIZ_EVERY_N = 5 # frequency to update visualization markers

    def __init__(self):
        super().__init__("rrt_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        traj_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 1)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.traj_pub = self.create_publisher(PoseArray, "/trajectory/current", traj_qos)
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_cb, 10)
        self.tree_a_pub = self.create_publisher(Marker, "/rrt/tree_a", 1)
        self.tree_b_pub = self.create_publisher(Marker, "/rrt/tree_b", 1)
        self.initialpose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initialpose_cb, 10)

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory", color=(0.0, 0.0, 1.0))

        self.map_info = None
        self.occupancy_grid = None
        self.free_cells = None
        self.car_pose = None

        self.get_logger().info("RRT-Connect planner initialized")

    def map_cb(self, msg):
        """store the map and free cells."""
        self.map_info = msg.info

        self.occupancy_grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.free_cells = np.argwhere(self.occupancy_grid == 0)

        o = msg.info.origin
        q = o.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.map_origin_x = o.position.x
        self.map_origin_y = o.position.y
        self.map_cos = np.cos(yaw)
        self.map_sin = np.sin(yaw)

        self.get_logger().info(f"Map received ")

    def world_to_grid(self, x, y):
        """world coords to grid coords."""
        dx = x - self.map_origin_x
        dy = y - self.map_origin_y
        mx = self.map_cos * dx + self.map_sin * dy
        my = -self.map_sin * dx + self.map_cos * dy
        u = int(np.floor(mx / self.map_info.resolution))
        v = int(np.floor(my / self.map_info.resolution))
        return u, v

    def grid_to_world(self, u, v):
        """grid coords to world coords."""
        mx = u * self.map_info.resolution
        my = v * self.map_info.resolution
        x = self.map_cos * mx - self.map_sin * my + self.map_origin_x
        y = self.map_sin * mx + self.map_cos * my + self.map_origin_y
        return x, y

    def is_free(self, x, y):
        """check if a point is free."""
        u, v = self.world_to_grid(x, y)
        if 0 <= u < self.map_info.width and 0 <= v < self.map_info.height:
            return self.occupancy_grid[v, u] == 0
        return False

    def collision_free(self, p1, p2):
        """check if the line segment is free."""
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 1e-6:
            return self.is_free(p1[0], p1[1])
        n_checks = max(2, int(dist / self.COLLISION_CHECK_RESOLUTION))
        for i in range(n_checks + 1):
            t = i / n_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not self.is_free(x, y):
                return False
        return True

    def pose_cb(self, msg):
        self.car_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,)

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
        self.plan_path(self.car_pose, goal)

    # ── RRT-Connect core ─────────────────────────────────────────────────

    def nearest(self, tree_points, point):
        """get the nearest node index."""
        diffs = tree_points[:len(tree_points)] - np.array(point)
        dists = np.einsum('ij,ij->i', diffs, diffs)
        return int(np.argmin(dists))

    def steer(self, from_pt, to_pt):
        """step toward a point by at most STEP_SIZE."""
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        dist = np.hypot(dx, dy)
        if dist <= self.STEP_SIZE:
            return to_pt
        ratio = self.STEP_SIZE / dist
        return (from_pt[0] + dx * ratio, from_pt[1] + dy * ratio)

    def extend(self, tree_points, tree_parents, point):
        """grow the tree toward a point."""
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
        """keep extending until it reaches or gets stuck."""
        while True:
            status, idx = self.extend(tree_points, tree_parents, point)
            if status != "advanced":
                return status, idx

    def extract_path(self, tree_a_points, tree_a_parents, tree_b_points, tree_b_parents, a_idx, b_idx):
        """build the path from both trees."""
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
        """sample a free cell in world coords."""
        if self.free_cells is None or len(self.free_cells) == 0:
            return None

        idx = np.random.randint(len(self.free_cells))
        v, u = self.free_cells[idx]
        return self.grid_to_world(u + 0.5, v + 0.5)

    def clear_current_plan(self):
        """clear the current plan."""
        self.trajectory.clear()
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()
        self.clear_trees()

    def shortcut_path(self, path):
        """try to shorten the path."""
        if len(path) <= 2:
            return path
        path = list(path)
        for _ in range(self.SHORTCUT_ITERATIONS):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            if self.collision_free(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path

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

    def get_min_obst_dist(self, path):
        """get the min path distance to obstacles."""
        min_dist = np.inf
        for pt in path:
            u, v = self.world_to_grid(pt[0], pt[1])
            for di in range(-20, 21):
                for dj in range(-20, 21):
                    nu, nv = u + di, v + dj
                    if 0 <= nu < self.map_info.width and 0 <= nv < self.map_info.height:
                        if self.occupancy_grid[nv, nu] != 0:
                            ox, oy = self.grid_to_world(nu, nv)
                            dist = np.hypot(pt[0] - ox, pt[1] - oy)
                            if dist < min_dist:
                                min_dist = dist
        inflation_m = 8 * 0.0504
        self.get_logger().info(f"min obstacle dist: {min_dist + inflation_m:.4f} m")

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

        # one tree from start, one from goal
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
            # pick a random free spot to grow toward
            q_rand = self.sample_free()
            if q_rand is None:
                continue

            # grow the active tree first
            status_a, idx_a = self.extend(tree_a_points, tree_a_parents, q_rand)
            if status_a != "trapped":
                new_pt = (tree_a_points[idx_a][0], tree_a_points[idx_a][1])

                # then try to connect the other tree to it
                status_b, idx_b = self.connect(tree_b_points, tree_b_parents, new_pt)

                if status_b == "reached":
                    # if they meet, build and publish the path
                    self.publish_tree(self.tree_a_pub, start_tree, start_parents, 1.0, 0.0, 0.0, 0)
                    self.publish_tree(self.tree_b_pub, goal_tree, goal_parents, 1.0, 0.0, 0.0, 1)

                    elapsed = time.time() - t_start
                    self.get_logger().info(f"planning time: {elapsed:.2f} s")

                    if swapped:
                        raw_path = self.extract_path(tree_b_points, tree_b_parents, tree_a_points, tree_a_parents, idx_b, idx_a)
                    else:
                        raw_path = self.extract_path(tree_a_points, tree_a_parents, tree_b_points, tree_b_parents, idx_a, idx_b)

                    path = self.shortcut_path(raw_path)

                    self.trajectory.clear()
                    for pt in path:
                        self.trajectory.addPoint(pt)
                    self.trajectory.update_distances()
                    self.get_logger().info(f"path length: {self.trajectory.distances[-1]:.2f} m")
                    self.get_min_obst_dist(path)

                    self.traj_pub.publish(self.trajectory.toPoseArray())
                    self.trajectory.publish_viz()
                    return

            # update the tree viz every few iters
            if i % self.VIZ_EVERY_N == 0:
                self.publish_tree(self.tree_a_pub, start_tree, start_parents, 1.0, 0.0, 0.0, 0)
                self.publish_tree(self.tree_b_pub, goal_tree, goal_parents, 1.0, 0.0, 0.0, 1)

            # swap so each tree gets a turn to lead
            tree_a_points, tree_b_points = tree_b_points, tree_a_points
            tree_a_parents, tree_b_parents = tree_b_parents, tree_a_parents
            swapped = not swapped

        elapsed = time.time() - t_start
        self.get_logger().error(f"RRT-Connect failed to find path after {self.MAX_ITERATIONS} iterations ({elapsed:.2f}s)")
        self.clear_current_plan()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
