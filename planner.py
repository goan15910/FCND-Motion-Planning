from enum import Enum, auto
from queue import PriorityQueue

import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import LineString
import networkx as nx
from sklearn.neighbors import KDTree

from planning_utils import A_star, Action, Action_3D, prune_path


class Plan(Enum):
    # Tier 0
    GRID = (auto(), 0)
    GRAPH = (auto(), 0)

    # Tier 1
    LINE = (auto(), 1)
    VOXEL = (auto(), 1)

    # Tier 2
    IDEAL = (auto(), 2)
    POTENTIAL = (auto(), 2)
    RRT = (auto(), 2)
     
    @property
    def level(self):
       return self.value[1]


class Planner:
    """
    Planner class for the drone.
    """
    def __init__(self,
                 filename,
                 rough=Plan.GRID,
                 local=Plan.LINE,
                 kinetics=Plan.IDEAL,
                 drone_alt=5,
                 safe_dist=3,
                 n_sample=300,
                 vox_size=5,
                 vox_shape=(5, 5, 5),
                 prune_rough=True,
                 prune_local=True):
        # Check plan
        assert (rough.level == 0) and \
               (local.level == 1) and \
               (kinetics.level == 2), \
               "Invalid plan"
        self.rough = rough
        self.local = local
        self.kinetics = kinetics

        # setup basic components
        self.fname = filename
        self.drone_alt = drone_alt
        self.safe_dist = safe_dist
        self.n_sample = n_sample
        self.vox_size = vox_size
        self.vox_shape = vox_shape
        self.prune_rough = True
        self.prune_local = True
        
        # plan util
        self.rough_path = None
        self.start = None
        self.goal = None


    @property
    def reach_goal(self):
        return len(self.rough_path) == 0


    def get_waypoints(self):
        "Return waypoints in (N, E, A, H) format"
        assert self.rough_path is not None, \
            "Run rough plan first"

        if self.reach_goal:
            return []

        local_path = self.local_plan(True)

        return local_path


    def rough_plan(self, start, goal):
        # setup mini-map data
        self.data = np.loadtxt(
                        self.fname,
                        delimiter=',',
                        dtype='Float64',
                        skiprows=3)
        self.set_map_info()

        # create graph from mini-map data
        self.rough_path = []
        if self.rough == Plan.GRID:
            self.graph = self.create_grid()
        elif self.rough == Plan.GRAPH:
            self.graph = self.create_graph(3)

        # create rough path
        start_2d = start[:2]
        goal_2d = goal[:2]
        path = None
        if self.rough == Plan.GRID:
            path = self.graph_a_star(start_2d, goal_2d, Action, self.prune_rough) 
            path = self.grid_to_NEDH(path)
        elif self.rough == Plan.GRAPH:
            path = self.graph_a_star(start_2d, goal_2d, None, self.prune_rough)

        # setup up plan util
        self.rough_path = path
        self.start = self.rough_path.pop(0)
        self.goal = self.rough_path.pop(0)


    def local_plan(self):
        path = []
        if self.local == Plan.LINE:
            path = self.line(self.start, self.goal)
        if self.local == Plan.VOXEL:
            path = self.vox_a_star(self.start, self.goal, self.prune_local)
        
        self.start = path[-1]
        self.goal = self.rough_path.pop(0)

        return path


    def grid_to_NEAH(self, path):
        new_path = []
        for p in path:
            new_p = [0, 0, 0, 0]
            new_p[0] = p[0] + self.n_min
            new_p[1] = p[1] + self.e_min
            new_p[2] = self.drone_alt
            new_path.append(new_p)
        return new_path


    def NEAH_to_voxel(self, path, offsets):
        n_min, e_min, a_min = offsets
        new_path = []
        for p in path:
            new_p = [0, 0, 0]
            new_p[0] = (p[0] - n_min) // self.vox_size
            new_p[1] = (p[1] - e_min) // self.vox_size
            new_p[2] = (p[2] - a_min) // self.vox_size
            new_path.append(new_p)
        return new_path


    def vox_to_NEAH(self, path, offsets):
        n_min, e_min, a_min = offsets
        new_path = []
        for p in path:
            new_p = [0, 0, 0, 0]
            new_p[0] = p[0]*self.vox_size + n_min
            new_p[1] = p[1]*self.vox_size + e_min
            new_p[2] = p[2]*self.vox_size + a_min
            new_path.append(new_p)
        return new_path


    def graph_a_star(self,
                     start,
                     goal,
                     actions,
                     prune):
        """Graph A* for rough plan"""
        a_star = A_star(self.graph, start, goal, actions)
        path = a_star.compute_path()
        
        if prune:
            path = prune_path(path, 1e-6)

        return path


    def vox_a_star(self, start, goal, prune):
        """Voxel A* for local plan"""
        voxmap, offsets = self.create_voxmap()
        vox_start, vox_goal = self.NEAH_to_vox([start, goal], offsets)
        a_star = A_star(voxmap, vox_start, vox_goal, Action_3D)
        path = a_star.compute_path()
        
        if prune:
            path = prune_path(path)

        path = self.vox_to_NEAH(path, offsets)

        return path


    def line(self, start, goal):
        """Move directly from start to goal"""
        return [start, goal]


    def set_map_info(self):
        data = self.data
        n_min = np.floor(np.min(data[:, 0] - data[:, 3]))
        n_max = np.ceil(np.max(data[:, 0] + data[:, 3]))
        e_min = np.floor(np.min(data[:, 1] - data[:, 4]))
        e_max = np.ceil(np.max(data[:, 1] + data[:, 4]))
        a_min = 0
        a_max = np.ceil(np.amax(data[:, 2] + data[:, 5]))

        # discretize the map data
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.e_min = int(e_min)
        self.e_max = int(e_max)
        self.a_min = int(a_min)
        self.a_max = int(a_max)
        self.n_size = int(np.ceil(self.n_max-self.n_min))
        self.e_size = int(np.ceil(self.e_max-self.e_min))
        self.a_size = int(np.ceil(self.a_max-self.a_min))


    def create_grid(self):
        """Create grid from mini-map data"""
        # Initialize
        grid = np.zeros((self.n_size, self.e_size))

        # Mark obstacles
        for i in range(self.data.shape[0]):
            n, e, a, d_n, d_e, d_a = self.data[i, :]

            if a + d_a + self.safe_dist > self.drone_alt:
                obstacle = [
                    int(np.clip(n - d_n - self.safe_dist - self.n_min, 0, self.n_size-1)),
                    int(np.clip(n + d_n + self.safe_dist - self.n_min, 0, self.n_size-1)),
                    int(np.clip(e - d_e - self.safe_dist - self.e_min, 0, self.e_size-1)),
                    int(np.clip(e + d_e + self.safe_dist - self.e_min, 0, self.e_size-1)),
                ]
                grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

        return grid


    def create_graph(self, k):
        # nodes and polygons
        self.extract_polygons()
        nodes = self.sample_nodes()

        # create graph
        g = nx.Graph()
        tree = KDTree(nodes)
        for n1 in nodes:
            idxs = tree.query([n1], k, return_distance=False)[0]

            for idx in idxs:
                n2 = nodes[idx]
                if n2 == n1:
                    continue

                if self.valid_edge(n1, n2):
                    dist = LA.norm(np.array(n1)-np.array(n2))
                    g.add_edge(n1, n2, weight=dist)

        return g


    def sample_nodes(self):
        num = self.n_sample
        nvals = np.random.uniform(self.n_min, self.n_max, num)
        evals = np.random.uniform(self.e_min, self.e_max, num)
        avals = np.random.uniform(self.a_min, self.a_max, num)

        nodes = np.hstack([
                    nvals[:, np.newaxis],
                    evals[:, np.newaxis],
                    avals[:, np.newaxis],
                ])

        centers = np.array([p.center for p in self.polys])
        tree = KDTree(centers, metric='euclidean')

        valid_nodes = []
        for n in nodes:
            _, idx = tree.query(np.array(n[:2]).reshape(1, -1))
            p = self.polys[int(idx)]
            if not p.contains(n) or p.height < n[2]:
                valid_nodes.append(n)

        return valid_nodes


    def extract_polygons(self):
        polys = []
        for i in range(self.data.shape[0]):
            n, e, a, d_n, d_e, d_a = data[i, :]
            corners = [
                (n - d_n, e - d_e),
                (n - d_n, e + d_e),
                (n + d_n, e + d_e),
                (n + d_n, e - d_e)]
            height = a + d_a
            p = Poly(corners, height)
            polys.append(p)
        self.polys = polys


    def valid_edge(self, n1, n2):
        l = LineString([n1, n2])
        for p in self.polys:
            min_h = min(n1[2], n2[2])
            if p.crosses(l) and p.height >= min_h:
                return False
        return True


    def create_voxmap(self):
        # basic parameters
        n_min_center = np.min(self.data[:, 0])
        e_min_center = np.min(self.data[:, 1])
        a_min_cetner = np.min(self.data[:, 2])
        offsets = (n_min_center, e_min_center, a_min_center)

        vox_shape = self.vox_shape
        vox_size = self.vox_size
        vox_start = self.NEAH_to_vox([self.start], offsets)[0]

        # create empty voxmap
        voxmap = np.zeros(vox_shape, dtype=np.bool)

        # determine if cell out of range
        def out_of_range(obs, h):
            n, e, a = vox_start
            d = vox_shape / 2
            if (obs[0] > n + d) or \
               (obs[1] < n - d) or \
               (obs[2] > e + d) or \
               (obs[3] < e - d) or \
               (h < a - d):
                return True 
            else:
                return False


        for i in range(self.data.shape[0]):
            n, e, a, d_n, d_e, d_a = self.data[i, :]
            obstacle = [
                int(n - d_n - n_min_center) // voxel_size,
                int(n + d_n - n_min_center) // voxel_size,
                int(e - d_e - e_min_center) // voxel_size,
                int(e + d_e - e_min_center) // voxel_size
            ]
            height = int(a + d_a) // voxel_size

            if out_of_range(obstacle, height):
                continue

            voxmap[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3], 0:height] = True
        
        return voxmap, (n_min_center, e_min_center, a_min_cetner)


    def viz_rough_path(self):
        #TODO
        pass


