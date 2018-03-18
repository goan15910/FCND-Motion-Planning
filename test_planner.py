
# coding: utf-8

from udacidrone.frame_utils import global_to_local
from planner import Planner


if __name__ == '__main__':

    # Setup home, start, goal
    home = [-122.397335, 37.792571, 0.]
    start = global_to_local(home, home)
    goal = global_to_local([-122.395788, 37.793772 , 0.], home)

    # Setup Planner
    planner = Planner("colliders.csv")
    planner.rough_plan(start, goal)

    path = []
    while not planner.reach_goal:
        points = planner.get_waypoints()
        path.extend(points)

    print("Number of waypoints {}".format(len(path)))

