from enum import Enum, auto
from queue import PriorityQueue

import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bresenham import bresenham

from planning_utils import A_star, prune_path, in_line, bump
from graph_maker import Grid_2d, Voronoi_2d, Voxel


OBSTACLES = 0
GRAPH = 1
LINE = 2
POINTS = 3


class Plan(Enum):
    # Tier 0
    GRID = (auto(), 0)
    VORONOI = (auto(), 0)

    # Tier 1
    SIMPLE = (auto(), 1)
    VOXEL = (auto(), 1)
     
    @property
    def level(self):
       return self.value[1]


class Planner:
    """
    Planner class for the drone.
    """
    def __init__(self,
                 filename,
                 raw=Plan.GRID,
                 local=Plan.SIMPLE,
                 drone_alt=5,
                 safe_dist=3,
                 vox_size=5,
                 vox_shape=(10, 10, 10),
                 prune_raw=True,
                 prune_local=True,
                 greedy_prune=False,
                 visualize=False,
                 verbose=False):
        # Check plan
        assert (raw.level == 0) and \
               (local.level == 1), \
               "Invalid plan"
        self.raw = raw
        self.local = local

        # setup basic components
        self.drone_alt = drone_alt
        self.safe_dist = safe_dist
        self.vox_size = vox_size
        self.vox_shape = vox_shape
        self.prune_raw = prune_raw
        self.prune_local = prune_local
        self.greedy_prune = greedy_prune
        self.visualize = visualize
        self.verbose = verbose
        
        # plan util
        self.grid = None # 2.5D grid
        self.centers = None # center of obstacles
        self.aux_2d_graph = None # 2D aux-graph
        self.aux_3d_graph = None # 3D aux-graph
        self.raw_path = None # raw 2D path
        self._start = None # very start
        self._goal = None # ultimate goal
        self._just_start = True

        # setup mini-map data
        self.data = np.loadtxt(
            filename,
            delimiter=',',
            dtype='Float64',
            skiprows=2)
        self._set_map_info()
        self._create_grid()


    @property
    def map_min(self):
        return self._min.astype(int)

    @property
    def map_max(self):
        return self._max.astype(int)

    @property
    def map_shape(self):
        return self._shape.astype(int)

    @property
    def start(self):
        return self.grid_NEAH([self._start])[0]

    @property
    def goal(self):
        return self.grid_NEAH([self._goal])[0]

    @property
    def raw_waypoints(self):
        return self.grid_to_NEAH(self.raw_path)

    @property
    def reach_goal(self):
        return len(self.raw_path) == 0

    @property
    def just_start(self):
        return self._just_start


    def NEAH_to_grid(self, path):
        "NEA(H) to 2.5 grid"
        new_path = []
        for p in path:
            new_p = np.array(p[:3]) - self.map_min
            new_path.append(tuple(map(int, new_p)))
        return new_path


    def grid_to_NEAH(self, path):
        "2.5 grid to NEA(H)"
        new_path = []
        for i,p in enumerate(path):
            new_p = np.zeros(4, dtype=float)
            new_p[:2] = np.array(p[:2]) + self.map_min[:2]
            new_p[2] = self.drone_alt
            if i != 0:
                p_prev = path[i-1]
            else:
                p_prev = new_p
            new_p[3] = np.arctan2(
                            (p[1]-p_prev[1]),
                            (p[0]-p_prev[0]))
            #new_path.append(tuple(map(float, new_p)))
            new_path.append(tuple(map(int, new_p)))
        return new_path


    def get_waypoints(self):
        "Return partial waypoints in NEAH format"
        assert self.raw_path is not None, \
            "Run raw plan first"

        waypoints = None
        if self.just_start:
            waypoints = self._refine_path()
            self._just_start = False
        elif len(self.raw_path) >= 2:
            waypoints = self._refine_path()[1:]
        elif self.reach_goal:
            if self.verbose:
                print("reach goal already")
            waypoints = []

        if self.verbose and not self.reach_goal:
            print("{} waypoints left".format(len(self.raw_path)))

        if len(waypoints) == 0:
            return []
        else:
            return self.grid_to_NEAH(waypoints)


    def _set_map_info(self):
        # discretize the map data
        data = self.data
        self._min = np.floor(np.min(data[:, :3] - data[:, 3:6], axis=0))
        self._max = np.ceil(np.max(data[:, :3] + data[:, 3:6], axis=0))
        self._shape = np.ceil(self._max-self._min)
        if self.verbose:
            print("min: ", self.map_min)
            print("max: ", self.map_max)
            print("shape: ", self.map_shape)


    def _create_grid(self):
        """Create 2.5D grid from mini-map data"""
        # Initialize
        if self.verbose:
            print("creating 2.5D grid ...")
        grid = np.zeros(self.map_shape[:2])
        centers = []

        # Mark obstacles
        for i in range(self.data.shape[0]):
            n, e, a, d_n, d_e, d_a = self.data[i, :]

            if a + d_a + self.safe_dist > self.drone_alt:
                obs = [
                    int(np.clip(n - d_n - self.safe_dist - self.map_min[0], 0, self.map_shape[0]-1)),
                    int(np.clip(n + d_n + self.safe_dist - self.map_min[0], 0, self.map_shape[0]-1)),
                    int(np.clip(e - d_e - self.safe_dist - self.map_min[1], 0, self.map_shape[1]-1)),
                    int(np.clip(e + d_e + self.safe_dist - self.map_min[1], 0, self.map_shape[1]-1)),
                ]
                height = int(a + d_a)
                grid[obs[0]:obs[1]+1, obs[2]:obs[3]+1] = height
                centers.append((int(n-self.map_min[0]), int(e-self.map_min[1])))

        self.grid = grid
        self.centers = centers


    def plan_path(self, start, goal):
        # convert start, goal to 2.5D grid format
        start,goal = self.NEAH_to_grid([start, goal])
        self._start = start
        self._goal = goal

        # Perform A* routing on indicated map graph
        self.raw_path = \
            self._compute_path(
                start,
                goal,
                self.raw,
                self.prune_raw)

        if self.verbose:
            print("{} raw waypoints".format(len(self.raw_path)))

        # visualize plan
        if self.visualize:
            self.viz_path(self.raw_waypoints)


    def _refine_path(self):
        # pop the start and goal of this trip
        start = self.raw_path.pop(0)
        goal = self.raw_path.pop(0)
        
        path = \
            self._compute_path(
                start,
                goal,
                self.local,
                self.prune_local)

        # set destination as the next start
        if not self.reach_goal:
            self.raw_path.insert(0, path[-1])

        return path


    def _compute_path(self,
                      start,
                      goal,
                      plan,
                      prune):
        # Select one plan
        if plan == Plan.GRID:
            map_graph = Grid_2d(
                self.grid,
                start,
                goal,
                self.drone_alt,
                self.safe_dist,
                verbose=self.verbose)
        elif plan == Plan.VORONOI:
            map_graph = Voronoi_2d(
                self.grid,
                self.centers,
                start,
                goal,
                self.drone_alt,
                self.safe_dist,
                verbose=self.verbose)
        elif plan == Plan.VOXEL:
            #TODO
            map_graph = None
            raise NotImplementedError
        elif plan == Plan.SIMPLE:
            map_graph = None

        # Compute path on the graph
        if map_graph is None:
            path = [start, goal]
        else:
            map_graph.create_graph()
            if plan == Plan.VORONOI:
                self.aux_2d_graph = map_graph.graph
            elif plan == Plan.VOXEL:
                self.aux_3d_graph = map_graph.graph
            a_star = A_star(map_graph,
                            verbose=self.verbose)
            path, cost = a_star.compute_path()

            if prune:
                path = self._prune_path(path)

        return path


    def _prune_path(self, path, epsilon=1e-6):
        #TODO
        if not self.greedy_prune:
            return prune_path(
                        path,
                        epsilon,
                        normalize=False)
        else:
            pruned_path = []
            pruned_path.append(path[0])
            idx = 2
            while idx < len(path):
                last = pruned_path[-1]
                cand = path[idx-1]
                p = path[idx]
                is_bump = bump(
                            last,
                            p,
                            self.grid,
                            self.drone_alt)
                if not is_bump:
                    idx += 1
                else:
                    pruned_path.append(cand)
                    idx += 1
            if pruned_path[-1] != path[-1]:
                pruned_path.append(path[-1])
            return pruned_path


    def viz_grid(self, binary, alt_offset):
        plt.rcParams['figure.figsize'] = 20, 20
        fig = plt.figure()
        grid = self.grid
        if alt_offset:
            grid = grid - self.drone_alt
            grid[np.where(grid < 0)] = 0
        if binary:
            grid[np.nonzero(grid)] = 1
        plt.imshow(
            grid,
            cmap='Greys',
            origin='lower',
            zorder=OBSTACLES)
        #plt.imshow(self.grid, cmap='Greys', origin='lower')


    def viz_graph(self, graph):
        """visualization for networkx graph"""
        for edge in graph.edges:
            p1 = edge[0]
            p2 = edge[1]
            plt.plot([p1[1], p2[1]],
                     [p1[0], p2[0]],
                     'green',
                     linewidth=4.0,
                     alpha=1.0,
                     zorder=GRAPH)


    def viz_path(self,
                 path,
                 graph=None,
                 draw_graph=True,
                 draw_line=True,
                 draw_points=True,
                 draw_end_points=True,
                 grid_binary=False,
                 grid_alt_offset=True):
        # visualize 2.5D/2D grid
        self.viz_grid(
            grid_binary,
            grid_alt_offset)

        # visualize graph
        if graph is None:
            graph = self.aux_2d_graph
        if draw_graph and (graph is not None):
            self.viz_graph(graph)

        # convert path to grid coords
        path = self.NEAH_to_grid(path)
    
        # draw path
        for i in range(len(path)):
            if draw_points:
                plt.scatter(path[i][1],
                            path[i][0],
                            c='red',
                            zorder=POINTS)

            if (i != len(path)-1) and draw_line:
                plt.plot([path[i][1], path[i+1][1]],
                         [path[i][0], path[i+1][0]],
                         'red',
                         linewidth=4.0,
                         alpha=1.0,
                         zorder=LINE)

        # draw start and goal
        if draw_end_points:
            plt.scatter(path[0][1],
                        path[0][0],
                        c='red',
                        s=40,
                        zorder=POINTS)
            plt.scatter(path[-1][1],
                        path[-1][0],
                        c='red',
                        s=40,
                        zorder=POINTS)

        plt.xlabel('NORTH')
        plt.ylabel('EAST')

        plt.show()


    def viz_vox_path(self):
        #TODO: visualize voxmap 3D path
        pass


