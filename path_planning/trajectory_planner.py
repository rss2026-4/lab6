import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import heapq as hp
from std_msgs.msg import Header
import numpy as np


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
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

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map = None
        self.map_dict = None
        self.map_received = 0
        self.start_pose = None
        self.goal_pose = None

        self.min_pub = self.create_publisher(MarkerArray, "/planned_trajectory/min_point_array", 1)
        

    def map_cb(self, msg):
            self.map = msg
            if self.map.data != msg.data or self.map_received == 0:
                self.get_logger().info("Creating Map Dict")

                # orientation extraction
                o = msg.info.origin
                q = o.orientation
                yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                self.map_origin_x = o.position.x
                self.map_origin_y = o.position.y
                self.map_cos = np.cos(yaw)
                self.map_sin = np.sin(yaw)

                self.map_dict = self.create_map_dict()
                self.map_received = 1
    
    def world_to_cell(self, x, y):
        dx = x - self.map_origin_x
        dy = y - self.map_origin_y

        grid_x = dx*self.map_cos + dy*self.map_sin
        grid_y = -dx*self.map_sin + dy*self.map_cos

        u = grid_x//self.map.info.resolution
        v = grid_y//self.map.info.resolution

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

    def create_map_dict(self):
        result = {}
        center = (self.map.info.origin.position.x, self.map.info.origin.position.y)

        self.get_logger().info("x %f " % center[0])
        self.get_logger().info("y %f " % center[1])
        self.get_logger().info("width %f " % self.map.info.width)
        self.get_logger().info("height %f " % self.map.info.height)
        self.get_logger().info("resolution %f " % self.map.info.resolution)
        self.get_logger().info(f"rotation = {self.map.info.origin.orientation} ")


        # x and y are cells/pixels of map; goal pose and start pose come in as meters
        for x in range(self.map.info.width):
            for y in range(self.map.info.height):
                indx = self.map.info.width * y + x # corresponding indx
                result[(x, y)] = self.map.data[indx]

        self.get_logger().info("Done! Map dict of size %d created" % len(result))
        return result

    def init_pose(self, msg):
        self.get_logger().info(f"clicked point: ({msg.pose.pose.position.x, msg.pose.pose.position.y})")
        self.start_pose = self.world_to_cell(msg.pose.pose.position.x, msg.pose.pose.position.y)

    def pose_cb(self, pose):
        pass

    def goal_cb(self, msg):
        # get coords in cell/grid frame
        self.goal_pose = self.world_to_cell(msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info("Start Pose: %s | End Pose: %s | map received: %s " % (self.start_pose, self.goal_pose, self.map_received))
        
        if self.map is not None:
            self.get_logger().info("Planning path!")
            self.plan_path(self.start_pose, self.goal_pose, self.map_received)
  
    def plan_path(self, start_point, end_point, map):
        traj = self.a_star_search(start_point, end_point)
        self.trajectory.points = traj
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.get_logger().info("Visualizing path")
        self.trajectory.publish_viz()
        if traj != []:
            self.get_min_obst_dist(traj)

    def get_dist(self, a, b):
        """
        takes in tuple pos (x, y) a and b
        and returns euclidean dist from a to b
        """
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return np.sqrt(dx**2 + dy**2)

    def heuristic(self, pos):
        """
        takes in tuple pos (x, y) and returns 
        euclidean dist to goal
        """
        return self.get_dist(pos, self.goal_pose)

    def cost_function(self, a, b):
        """
        takes in tuple pos (x, y) a and b
        and returns euclidean dist from a to b
        """
        return self.get_dist(a, b)
    
    def a_star_search(self, start, end, heuristic=heuristic, cost=cost_function):       
        cost_so_far = {start: 0} # dict where keys correspond to locs and values are cost to get there from start
        came_from = {start: None}
        frontier = []
        hp.heapify(frontier)
        hp.heappush(frontier, (0, start)) #add start to priority queue with lowest cost
        start_time = self.get_clock().now().nanoseconds / 1e9
        while frontier:
            # self.get_logger().info("Planning")
            # path = frontier.pop(0)
            current_pos = hp.heappop(frontier)[1]
            # self.get_logger().info(f"{current_pos=}")
            
            
            if current_pos == end:
                # self.get_logger().info("%s" % end)
                end_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info(f"Goal Reached! Path Planned in {end_time-start_time} seconds")
                return self.extract_path(came_from)
            
            for neighbor in self.get_neighbors(current_pos):
                # self.get_logger().info("visiting neighbor: %s " % neighbor)
                new_cost = cost_so_far[current_pos] + self.cost_function(current_pos, neighbor)
                
                # if neighbor not yet explord or cost to go there is less than before
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor)
                    hp.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current_pos


        self.get_logger().info("No Path Found")
        return []

    def extract_path(self, node_dict):
        """
        returns a path as list of (x, y) tuples
        given dictionary node_dict where keys correspond to 
        the position that comes from their value

        example input: 
        node_dict{
            (start): None
            (x2, y2): (x1, y1)
            (x3, y3): (x2, y2)
        }
        """
        
        path = [self.grid_to_world(*self.goal_pose)]

        past_node = self.goal_pose
        while past_node != self.start_pose:
            next_node = node_dict[past_node]
            path.append(self.grid_to_world(*next_node))
            past_node = next_node
        
        return path[::-1]

    def get_neighbors(self, current_position):
        """
        return 3x3 grid of neighbors 
        with current position at center
        """
        # self.get_logger().info("getting neighbors")
        neighbors = set()
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_pos = (current_position[0]+i, current_position[1]+j)
                # self.get_logger().info(f"new neighbor: {new_pos}")
                if new_pos not in neighbors and self.is_valid(new_pos):
                    # self.get_logger().info(f"new neighbor added: {new_pos}")
                    neighbors.add(new_pos)
        return neighbors

    def is_valid(self, pos, occup_threshold=50):
        pos = (int(pos[0]), int(pos[1]))
        if pos[0] < 0 or pos[1] < 0:
            return False

        # unknown space invalid
        if self.map_dict[pos] == -1:
            return False
        
        # obstacles invalid
        if self.map_dict[pos] >= occup_threshold:
            return False
        
        return True

    def get_obstacles(self, occup_threshold=0.5):
        obstacles = set()
        for point in self.map_dict:
            if self.map_dict[point] >= occup_threshold:
                obstacles.add(point)
        self.get_logger().info("all obstacles found")
        
        return obstacles

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

            
        self.get_logger().info(f"Min at path point {min_path_point}, nearest obstacle at {min_dist_point} with dist: {min_dist} m")

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
