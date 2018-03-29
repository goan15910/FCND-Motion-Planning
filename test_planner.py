
# coding: utf-8

from udacidrone.frame_utils import global_to_local
from planner import Planner, Plan


if __name__ == '__main__':

    # Setup home, start, goal
    home = [-122.397335, 37.792571, 0.]
    cands = {
        "market st": [-122.395788, 37.793772, 0.],
        "drum st": [-122.396385, 37.795124, 0.],
        "front st": [-122.398925, 37.792702, 0.],
        "clay-davis cross": [-122.398249, 37.796079, 0.],
    }
    start = global_to_local(home, home)
    goal = global_to_local(cands["market st"], home)

    # Setup Planner
    planner = Planner("colliders.csv",
                      rough=Plan.GRID,
                      local=Plan.LINE,
                      drone_alt=5,
                      safe_dist=3,
                      n_sample=200,
                      n_neighbors=10,
                      prune_rough=False,
                      verbose=True)
    planner.rough_plan(start, goal)

    path = []
    while not planner.reach_goal:
        points = planner.get_waypoints()
        path.extend(points)

    print("Path details:")
    print("{} waypoints".format(len(path)))

    planner.viz_path(path)



