import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import numpy.linalg as LA

from planning_utils import a_star, heuristic, create_grid, prune_path
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def displacement(self, degree):
        return self.target_position[:degree] - self.local_position[:degree]

    def arrived(self, tolerance, degree=2):
        """Arrived at target or not"""
        return LA.norm(self.displacement(degree)) < tolerance

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if self.arrived(1.0, degree=2):
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if LA.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position {}'.format(self.target_position))
        self.cmd_position(*self.target_position)

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 10 # original value: 3

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        with open("colliders.csv") as f:
            top_line = f.readline().strip()
            coord = top_line.replace("lat0 ", "").replace("lon0", "").split(", ")
            coord = list(map(float, coord))
        
        # TODO: set home position to (lat0, lon0, 0)
        self.set_home_position(coord[1], coord[0], 0)

        # TODO: retrieve current global position
        # TODO: convert to current local position using global_to_local()
        current_local_pos = global_to_local(self.global_position, self.global_home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home,
                                                                         self.global_position,
                                                                         self.local_position))
        print("current local position {}".format(current_local_pos))

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=3)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        print("Grid shape: {}".format(grid.shape))
        # Define starting point on the grid (this is just grid center)
        start = (int(current_local_pos[0]-north_offset), int(current_local_pos[1]-east_offset))
        
        # Set goal as some arbitrary position on the grid
        home = global_to_local([-122.397335, 37.792571, 0], self.global_home)
        market_st = global_to_local([-122.395788, 37.793772, 0.], self.global_home)
        drum_st = global_to_local([-122.396385, 37.795124, 0.], self.global_home)
        front_st = global_to_local([-122.398925, 37.792702, 0.], self.global_home)
        #clay_davis_cross = global_to_local([-122.398249, 37.796079, 0.], self.global_home)
        my_goal = drum_st
        goal = (int(my_goal[0]-north_offset), int(my_goal[1]-east_offset))

        # Run A* to find a path from start to goal
        # TODO: (done) add diagonal motions with a cost of sqrt(2) to your A* implementation
        # TODO: or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', start, goal)
        path, _ = a_star(grid, heuristic, start, goal)
        
        # TODO: prune path to minimize number of waypoints
        path = prune_path(path, epsilon=1e-6)
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        print("Number of waypoints: {}".format(len(waypoints)))
        # TODO: send waypoints to sim
        #self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        #while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
