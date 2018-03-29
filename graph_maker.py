import numpy as np
import numpy.linalg as LA
from bresenham import bresenham
from scipy.spatial import Voronoi, voronoi_plot_2d
import networkx as nx

from planning_utils import Map_graph, Action, Action_3D, valid_next_nodes, valid_cell, bump


class Grid_2d(Map_graph):
    """
    Create 2D grid.
    """
    def __init__(self,
                 grid,
                 start,
                 goal,
                 drone_alt,
                 safe_dist,
                 verbose=False):
        Map_graph.__init__(
            self,
            grid,
            start[:2],
            goal[:2],
            actions=Action,
            verbose=verbose)
        self.drone_alt = drone_alt
        self.safe_dist = safe_dist


    def create_graph(self):
        """Create grid from mini-map data"""        
        # Make 2D graph from 2.5D grid
        if self.verbose:
            print("creating 2D grid ...")
        obs_idxs = np.where(self.grid > self.drone_alt + self.safe_dist)
        self.graph = np.zeros_like(self.grid)
        self.graph[obs_idxs] = 1


    def next_nodes(self, n1, n1_cost):
        nodes = []
        for n2,a in valid_next_nodes(n1, self.graph, self.actions):
            n2_cost = n1_cost + a.cost + self.h_func(n2, self.goal)
            nodes.append((n2, n2_cost))
        return nodes


class Voronoi_2d(Map_graph):
    """
    Create Voronoi 2D graph
    """
    def __init__(self,
                 grid,
                 centers,
                 start,
                 goal,
                 drone_alt,
                 safe_dist,
                 connect_graph=True,
                 k=10,
                 verbose=False):
        Map_graph.__init__(
            self,
            grid,
            start[:2],
            goal[:2],
            actions=Action,
            verbose=verbose)
        self.centers = centers
        self.drone_alt = drone_alt
        self.safe_dist = safe_dist
        self.search_k = k
        self.connect_graph = connect_graph


    def create_graph(self):
        v_graph = Voronoi(self.centers)
        grid = self.grid
        self.graph = nx.Graph()

        # test if collide with obstacles
        def collision(p1, p2):
            cells = list(bresenham(p1[0], p1[1], p2[0], p2[1]))
            hit = False
            for c in cells:
                # First check if we're off the map
                if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                    hit = True
                    break
                # Next check if we're in collision
                if grid[c[0], c[1]] >= self.drone_alt:
                    hit = True
                    break
            return hit

        # construct nx graph
        for v in v_graph.ridge_vertices:
            p1 = v_graph.vertices[v[0]]
            p2 = v_graph.vertices[v[1]]
            p1 = tuple(map(int, p1[:2]))
            p2 = tuple(map(int, p2[:2]))

            # add valid node and edge to graph
            if not bump(p1, p2, grid, self.drone_alt):
                self.graph.add_node(p1)
                self.graph.add_node(p2)
                self.graph.add_edge(p1, p2)

        if self.connect_graph:
            sub_g_list = sorted(
                nx.connected_components(self.graph),
                key=len, 
                reverse=True)
            self.graph = self.graph.subgraph(sub_g_list[0]).copy()

        # add start and goal
        for p in [self.start, self.goal]:
            try:
                self.graph[p]
            except KeyError:
                n_list = []
                for n in self.graph.nodes:
                    d = LA.norm(np.array(p)-np.array(n))
                    n_list.append((n, d))
                cands = sorted(n_list, key=lambda x: x[1])[:self.search_k]
                found = False
                for n,d in cands:
                    if not bump(n, p, grid, self.drone_alt):
                        self.graph.add_node(p)
                        self.graph.add_edge(p, n)
                        found = True
                        break
                if not found:
                    raise ValueError("Can't add {} to graph".format(n))


    def next_nodes(self, n1, n1_cost):
        n2_list = []
        for n2 in self.graph[n1]:
            a_cost = LA.norm(np.array(n1)-np.array(n2))
            n2_cost = n1_cost + a_cost + self.h_func(n2, self.goal)
            n2_list.append((n2, n2_cost))
        return n2_list
        

# TODO

class Voxel(Map_graph):
    """
    Create 3D voxel map.
    """
    def __init__(self,
                 grid,
                 start,
                 goal,
                 vox_size,
                 vox_shape,
                 verbose=False):
        Map_graph.__init__(
            self,
            grid,
            start,
            goal,
            actions=Action_3D,
            verbose=verbose)
        self.vox_size = vox_size
        self.vox_shape = vox_shape


    def NEAH_to_voxel(self, path):
        offsets = self.offsets
        new_path = []
        for p in path:
            new_p = (np.array(p)-np.array(offsets)) // self.vox_size
            new_path.append(tuple(new_p.astype(np.int64)))
        return new_path


    def NEAH_path(self, path):
        offsets = self.offsets
        new_path = []
        for p in path:
            new_p = np.zeros(4, dtype=np.int64)
            new_p[:3] = np.array(p)*self.vox_size + np.array(offsets)
            new_path.append(tuple(new_p.astype(np.int64)))
        return new_path


    def create_graph(self):
        # basic parameters
        offsets = np.min(self.data[:, :2])

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

        #TODO
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


    def next_nodes(self, n1, n1_cost):
        pass