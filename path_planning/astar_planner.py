import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray
import heapq as hp
from std_msgs.msg import Header
import numpy as np
from scipy.ndimage import distance_transform_edt


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("astar_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "/map")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        # Cells within this radius of an obstacle get a clearance penalty
        # that grows linearly as you approach the wall. Larger radius = path
        # is pulled further from walls; cells beyond have zero penalty.
        self.clearance_radius = 25  # cells
        self.clearance_weight = 5.0  # max penalty per cell at the wall
        # Weighted A*: priority = g + heur_weight * h. >1 trades optimality
        # for fewer expansions; 1.0 = optimal.
        self.heur_weight = 2.0

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            latched_qos)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/astar/trajectory",
            10
        )

        self.pose_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_cb,
            10
        )

        self.initpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.init_pose,
            1
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/astar/planned_trajectory", color=(0.0, 1.0, 0.0))

        self.map = None
        self.map_arr = None
        self.obstacle_costs = None
        self.map_received = 0
        self.start_pose = None
        self.goal_pose = None

        self.min_pub = self.create_publisher(MarkerArray, "/astar/planned_trajectory/min_point_array", 1)
        
# ── Map Helpers ─────────────────────────────────────────────────
    def map_cb(self, msg):
            self.get_logger().info("map recieved")
            self.map = msg
            if self.map.data != msg.data or self.map_received == 0:
                # orientation extraction
                o = msg.info.origin
                q = o.orientation
                yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                self.map_origin_x = o.position.x
                self.map_origin_y = o.position.y
                self.map_cos = np.cos(yaw)
                self.map_sin = np.sin(yaw)

                self.map_arr = np.asarray(msg.data, dtype=np.int16).reshape(
                    msg.info.height, msg.info.width
                )
                self.obstacle_costs = self.create_obstacle_costs()
                self.map_received = 1
                self.get_logger().info(
                    f"Calculated obstacle cost lookup table with {int(np.count_nonzero(self.obstacle_costs))} penalized cells"
                )
    
    def world_to_cell(self, x, y):
        dx = x - self.map_origin_x
        dy = y - self.map_origin_y

        grid_x = dx*self.map_cos + dy*self.map_sin
        grid_y = -dx*self.map_sin + dy*self.map_cos

        u = int(grid_x // self.map.info.resolution)
        v = int(grid_y // self.map.info.resolution)

        return (u, v)
    
    def grid_to_world(self, u, v):
        """
        Convert grid coords to world coords
        """
        mx = u * self.map.info.resolution
        my = v * self.map.info.resolution
        x = self.map_cos * mx - self.map_sin * my + self.map_origin_x
        y = self.map_sin * mx + self.map_cos * my + self.map_origin_y
        return x, y

    def create_obstacle_costs(self, occup_threshold=50):
        # Exact Euclidean distance (in cells) from each free cell to the
        # nearest obstacle/unknown cell. One scipy call instead of a Python
        # offset loop.
        obstacle_mask = (self.map_arr >= occup_threshold) | (self.map_arr == -1)
        dist = distance_transform_edt(~obstacle_mask)

        # Linear ramp: max penalty at the wall, zero past clearance_radius.
        cost_grid = self.clearance_weight * np.maximum(
            0.0, self.clearance_radius - dist
        )
        return cost_grid.astype(np.float32)

# ── Planning Helpers ─────────────────────────────────────────────────
    def init_pose(self, msg):
        if self.map is None:
            self.get_logger().warn("No map received yet, cannot set initial pose")
            return
        self.start_pose = self.world_to_cell(msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.get_logger().info(
            f"Received start point: world=({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f}), "
            f"cell={self.start_pose}"
        )
        self.trajectory.clear()
        self.trajectory.publish_viz()

    def pose_cb(self, pose):
        pass

    def goal_cb(self, msg):
        if self.map is None:
            self.get_logger().warn("No map received yet, cannot plan")
            return
        if self.start_pose is None:
            self.get_logger().warn("Set an initial pose before sending a goal")
            return
        self.goal_pose = self.world_to_cell(msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f"Received goal point: world=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}), "
            f"cell={self.goal_pose}"
        )
        self.plan_path(self.start_pose, self.goal_pose, self.map_received)
  
    def plan_path(self, start_point, end_point, map):
        self.clear_min_vis()
        self.get_logger().info(f"Planning path from {start_point} to {end_point}")
        cell_path = self.a_star_search(start_point, end_point)
        traj = [self.grid_to_world(*c) for c in cell_path]
        self.trajectory.clear()
        self.trajectory.points = traj
        self.trajectory.update_distances()
        if traj == []:
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
            return

        self.get_logger().info(f"path length: {self.trajectory.distances[-1]:.2f} m")
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()
        # get_min_obst_dist is useful for debugging, but slow enough to affect planner timing.
        # if traj != []:
        #     self.get_min_obst_dist(traj)

# ── A* Functions ─────────────────────────────────────────────────

    def get_dist(self, a, b):
        """
        takes in tuple pos (x, y) a and b
        and returns euclidean dist from a to b
        """
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return np.sqrt(dx**2 + dy**2)

    def a_star_search(self, start, end):
        # Hot-loop locals: avoid attribute lookups inside the inner loop.
        map_arr = self.map_arr
        cost_grid = self.obstacle_costs
        h, w = map_arr.shape
        occ_thr = 50
        SQRT2 = 1.41421356237
        heur_w = self.heur_weight
        heappush = hp.heappush
        heappop = hp.heappop
        sx, sy = start
        gx, gy = end

        # 8-connected neighbor offsets with their step costs.
        NEIGHBORS = (
            (-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
            ( 0, -1, 1.0),                 ( 0, 1, 1.0),
            ( 1, -1, SQRT2), ( 1, 0, 1.0), ( 1, 1, SQRT2),
        )

        # g-scores and closed flags as 2D arrays — O(1) array indexing
        # beats dict hashing on hot path. Unvisited cells stay at +inf.
        g = np.full((h, w), np.inf, dtype=np.float32)
        g[sy, sx] = 0.0
        closed = np.zeros((h, w), dtype=bool)
        came_from = {start: None}
        frontier = [(0.0, start)]
        start_time = self.get_clock().now().nanoseconds / 1e9

        while frontier:
            _, current_pos = heappop(frontier)
            cx, cy = current_pos
            if closed[cy, cx]:
                continue
            closed[cy, cx] = True

            if current_pos == end:
                end_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info(f"planning time: {end_time-start_time:.2f} s")
                path = self.extract_path(came_from)
                self.get_logger().info(f"Trajectory found with {len(path)} points")
                return path

            cur_g = g[cy, cx]

            for di, dj, step in NEIGHBORS:
                nx, ny = cx + di, cy + dj
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if closed[ny, nx]:
                    continue
                val = map_arr[ny, nx]
                if val == -1 or val >= occ_thr:
                    continue

                new_cost = cur_g + step + cost_grid[ny, nx]
                if new_cost < g[ny, nx]:
                    g[ny, nx] = new_cost
                    # Octile distance: exact min path on 8-connected grid
                    # without obstacles. Tighter than Euclidean -> fewer expansions.
                    hdx = abs(nx - gx)
                    hdy = abs(ny - gy)
                    h_est = (hdx + hdy) + (SQRT2 - 2) * min(hdx, hdy)
                    priority = new_cost + heur_w * h_est
                    neighbor = (nx, ny)
                    heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current_pos

        self.get_logger().info("No Path Found")
        return []

    def extract_path(self, node_dict):
        """
        Walk came_from back from goal to start. Returns a list of cell
        tuples (x, y) ordered start -> goal.
        """
        path = [self.goal_pose]
        past_node = self.goal_pose
        while past_node != self.start_pose:
            next_node = node_dict[past_node]
            path.append(next_node)
            past_node = next_node
        return path[::-1]

    def get_obstacles(self, occup_threshold=50):
        ys, xs = np.where(self.map_arr >= occup_threshold)
        return set(zip(xs.tolist(), ys.tolist()))

    def make_header(self, frame_id, stamp=None):
        if stamp is None:
            stamp = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header

    def get_min_obst_dist(self, path):
        min_dist = np.inf
        min_dist_point = None
        min_path_point = None
        obstacles = self.get_obstacles()
        
        # ik its not that efficient but whatevs, its not meant for real time execution anyways
        for point in path:
            point_cell = self.world_to_cell(point[0], point[1])
            for i in range(-20, 20):
                for j in range(-20, 20):
                    candidate = (point_cell[0]+i, point_cell[1]+j)
                    
                    if candidate in obstacles:
                        cand_real = self.grid_to_world(*candidate)
                        dist = self.get_dist(point, cand_real)
                        
                        if dist < min_dist:
                            min_dist_point = cand_real
                            min_path_point = point
                            min_dist = dist                            
                            self.pub_min_vis(min_path_point, min_dist_point)

            
        inflation_m = 8 * 0.0504
        self.get_logger().info(f"min obstacle dist: {min_dist + inflation_m:.4f} m")

    def clear_min_vis(self):
        markers = MarkerArray()
        for i in range(2):
            m = Marker()
            m.header = self.make_header("/map")
            m.ns = "/planned_trajectory/trajectory"
            m.id = i
            m.action = Marker.DELETE
            markers.markers.append(m)
        self.min_pub.publish(markers)

    def pub_min_vis(self, path_point, min_point):
        "takes in point in world frame"
        points = MarkerArray() 

        min_marker = Marker()
        min_marker.header = self.make_header("/map")
        min_marker.ns = "/planned_trajectory/trajectory"
        min_marker.id = 0
        min_marker.type = 2  # sphere
        min_marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        min_marker.action = 0
        min_marker.pose.position.x = min_point[0]
        min_marker.pose.position.y = min_point[1]
        min_marker.pose.orientation.w = 1.0
        min_marker.scale.x = 1.0
        min_marker.scale.y = 1.0
        min_marker.scale.z = 1.0
        min_marker.color.r = 0.0
        min_marker.color.g = 1.0
        min_marker.color.b = 1.0
        min_marker.color.a = 1.0

        path_marker = Marker()
        path_marker.header = self.make_header("/map")
        path_marker.ns = "/planned_trajectory/trajectory"
        path_marker.id = 1
        path_marker.type = 2  # sphere
        path_marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        path_marker.action = 0
        path_marker.pose.position.x = path_point[0]
        path_marker.pose.position.y = path_point[1]
        path_marker.pose.orientation.w = 1.0
        path_marker.scale.x = 1.0
        path_marker.scale.y = 1.0
        path_marker.scale.z = 1.0
        path_marker.color.r = 0.5
        path_marker.color.g = 1.0
        path_marker.color.b = 0.5
        path_marker.color.a = 1.0

        points.markers.append(min_marker)
        points.markers.append(path_marker)
        
        self.min_pub.publish(points)


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
