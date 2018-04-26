import sys

import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import numpy.linalg as LA

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local
from planner import Plan, Planner
from planning_utils import read_global_home


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
            if -1.0 * self.local_position[2] > 0.95 * self.drone_alt:
                print("start waypoint trans!")
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if self.arrived(1.0, degree=2):
                if not (self.planner.reach_goal and len(self.waypoints) == 0):
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
        self.takeoff(self.drone_alt)

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        self.target_position = self.waypoints.pop(0)
        print('target position {}'.format(self.target_position))
        self.cmd_position(*self.target_position)
        self.get_waypoints()

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
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def send_raw_waypoints(self):
        data = msgpack.dumps(self.planner.raw_waypoints)
        self.connection._master.write(data)

    def get_waypoints(self):
        if len(self.waypoints) == 0:
            next_waypoints = self.planner.get_waypoints()
            self.waypoints.extend(next_waypoints)
            print(self.waypoints)


    def set_task(self,
                 global_goal,
                 raw=Plan.GRID,
                 local=Plan.SIMPLE,
                 drone_alt=5,
                 safe_dist=3,
                 prune_raw=True,
                 prune_local=True,
                 visualize=False,
                 verbose=False,
                 **kwargs):
        self.global_goal = global_goal
        self.planner = Planner("colliders.csv",
                               raw=raw,
                               local=local,
                               drone_alt=drone_alt,
                               safe_dist=safe_dist,
                               prune_raw=prune_raw,
                               prune_local=prune_local,
                               visualize=visualize,
                               verbose=verbose,
                               **kwargs)
        self.drone_alt = drone_alt
        self.safe_dist = safe_dist


    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")

        # TODO: read lat0, lon0 from colliders into floating point values
        coord = read_global_home("colliders.csv")
        
        # TODO: set home position to (lat0, lon0, 0)
        self.set_home_position(*coord)

        # TODO: retrieve current global position
        # TODO: convert to current local position using global_to_local()
        current_local_pos = global_to_local(self.global_position, self.global_home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home,
                                                                         self.global_position,
                                                                         self.local_position))
        print("current local position {}".format(current_local_pos))

        start = current_local_pos
        goal = global_to_local(self.global_goal, self.global_home)

        self.planner.plan_path(start, goal)
        self.send_raw_waypoints()
        self.get_waypoints()


    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        #while self.in_mission:
        #    pass

        self.stop_log()


# Candidate locations
cands = {
    "home": [-122.397450, 37.792480, 0],
    "market st": [-122.395788, 37.793772, 0.],
    "drum st": [-122.396385, 37.795124, 0.],
    "roof near drum st": [-122.396675, 37.794832, 0.],
    "front st": [-122.398925, 37.792702, 0.],
    "clay-davis cross": [-122.398249, 37.796079, 0.],
    "clay-davis forest": [-122.398087, 37.796235, 0.],
    "washington st forest": [-122.396553, 37.797314, 0.],
    "flat mall": [-122.395127, 37.792985, 0.],
    "fremont st": [-122.397800, 37.790606, 0.],
    "fremont alley": [-122.398470, 37.790556, 0.],
    "bush-market cross": [-122.399595, 37.790689, 0.],
    "sutter-market cross": [-122.400839, 37.789681, 0.],
    "front-california cross": [-122.399220, 37.793537, 0.],
    "sacramento st": [-122.399008, 37.794815, 0.],
    "washington-battery cross": [-122.401233, 37.796779, 0.],
    "the embarcadero": [-122.393771, 37.796983, 0.],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--goal', type=str, default='market st', help='goal location name')
    parser.add_argument('--verbose', help="increase output verbosity", default=False, action="store_true")
    parser.add_argument('--list', help="list available locations", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available locations:")
        for name in cands.keys():
            print(name)
        sys.exit()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    goal = cands[args.goal]
    print("Goal location: ", args.goal)

    TARGET_ALTITUDE = 5
    SAFETY_DISTANCE = 3 # original value: 3

    drone.set_task(
        goal,
        raw=Plan.VORONOI,
        local=Plan.SIMPLE,
        drone_alt=TARGET_ALTITUDE,
        safe_dist=SAFETY_DISTANCE,
        prune_raw=True,prune_local=False,
        greedy_prune=True,
        visualize=False,
        verbose=args.verbose)

    drone.start()
