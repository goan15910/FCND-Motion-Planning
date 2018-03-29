from enum import Enum
from queue import PriorityQueue

import numpy as np
import numpy.linalg as LA

from bresenham import bresenham

from shapely.geometry import Polygon, Point, LineString


def read_global_home(filename):
    with open(filename, 'r') as f:
        top_line = f.readline().strip()
        coord = top_line.replace("lat0 ", "").replace("lon0", "").split(", ")
        coord = list(map(float, coord))
    return [coord[1], coord[0], 0]


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)
    NORTH_EAST = (-1, 1, int(np.sqrt(2)))
    SOUTH_EAST = (1, 1, int(np.sqrt(2)))
    SOUTH_WEST = (1, -1, int(np.sqrt(2)))
    NORTH_WEST = (-1, -1, int(np.sqrt(2)))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


class Action_3D(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 3 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """
    W = (0, -1, 0, 1)
    E = (0, 1, 0, 1)
    N = (-1, 0, 0, 1)
    S = (1, 0, 0, 1)
    NE = (-1, 1, 0, int(np.sqrt(2)))
    SE = (1, 1, 0, int(np.sqrt(2)))
    SW = (1, -1, 0, int(np.sqrt(2)))
    NW = (-1, -1, 0, int(np.sqrt(2)))
    
    WU = (0, -1, -1, int(np.sqrt(2)))
    EU = (0, 1, -1, int(np.sqrt(2)))
    NU = (-1, 0, -1, int(np.sqrt(2)))
    SU = (1, 0, -1, int(np.sqrt(2)))
    NEU = (-1, 1, -1, int(np.sqrt(3)))
    SEU = (1, 1, -1, int(np.sqrt(3)))
    SWU = (1, -1, -1, int(np.sqrt(3)))
    NWU = (-1, -1, -1, int(np.sqrt(3)))
    
    WD = (0, -1, 1, int(np.sqrt(2)))
    ED = (0, 1, 1, int(np.sqrt(2)))
    ND = (-1, 0, 1, int(np.sqrt(2)))
    SD = (1, 0, 1, int(np.sqrt(2)))
    NED = (-1, 1, 1, int(np.sqrt(3)))
    SED = (1, 1, 1, int(np.sqrt(3)))
    SWD = (1, -1, 1, int(np.sqrt(3)))
    NWD = (-1, -1, 1, int(np.sqrt(3)))

    @property
    def cost(self):
        return self.value[3]

    @property
    def delta(self):
        return self.value[:3]


class Map_graph:
    """
    Map graph base class. 
    Convert 2.5D grid to customized graph.
    """
    def __init__(self,
                 grid,
                 start,
                 goal,
                 actions=None,
                 verbose=False):
        # basic parameters
        self.grid = grid
        self.graph = None
        self.actions = actions
        self.h_func = heuristic
        self._start = start
        self._goal = goal
        self.verbose = verbose

    @property
    def start(self):
        return tuple(map(int, self._start))

    @property
    def goal(self):
        return tuple(map(int, self._goal))

    def create_graph(self):
        raise NotImplementedError

    def next_nodes(self, n1, n1_cost):
        raise NotImplementedError


class A_star:
    """
    A star algorithm for 2D grid / 3D grid / graph
    """
    def __init__(self,
                 map_graph,
                 verbose=False):
        # setup basic component of A*
        self.map = map_graph
        self.start = self.map.start
        self.goal = self.map.goal
        self.verbose = verbose

        if self.verbose:
            print("start: ", self.start)
            print("goal: ", self.goal)


    def compute_path(self):
        path = []
        path_cost = 0
        queue = PriorityQueue()
        queue.put((0, self.start))
        visited = set(self.start)

        branch = {}
        found = False

        while not queue.empty():
            item = queue.get()
            n1_cost = item[0]
            n1 = item[1]

            if n1 == self.goal:
                if self.verbose:
                    print('Found a path.')
                found = True
                break
            else:
                n2_list = self.map.next_nodes(n1, n1_cost)
                for n2,n2_cost in n2_list:
                    if n2 not in visited:
                        visited.add(n2)
                        queue.put((n2_cost, n2))
                        branch[n2] = (n2_cost, n1)
        
        if found:
            n = self.goal
            path_cost = branch[n][0]
            path.append(self.goal)
            while branch[n][1] != self.start:
                path.append(branch[n][1])
                n = branch[n][1]
            path.append(branch[n][1])
        else:
            print('**********************')
            print('Failed to find a path!')
            print('**********************') 

        return path[::-1], path_cost


def heuristic(node, goal):
    node = np.array(node)
    goal = np.array(goal)
    return LA.norm(node-goal)


def valid_next_nodes(cur_node, graph, actions):
    assert actions is not None, \
            "Need to specify actions!"
    all_actions = list(actions)
    valid_actions = []
    valid_nodes = []
    for a in all_actions:
        mapped =  map(sum, zip(cur_node, a.delta))
        next_node = tuple(mapped)
        if valid_cell(next_node, graph):
            valid_nodes.append(next_node)
            valid_actions.append(a)
    return list(zip(valid_nodes, valid_actions))


def valid_cell(node, graph):
    shape = graph.shape
    n_dim = len(shape)
    node = tuple(map(int, node))
    for i in range(n_dim):
        if (node[i] < 0) or (node[i] > shape[i]-1):
            return False
        elif graph[node] == 1:
            return False
    return True    


def bump(p1, p2, grid, alt_offset):
    # bresenham algorithm
    cells = list(bresenham(p1[0], p1[1], p2[0], p2[1]))
    hit = False
    for c in cells:
        # First check if we're off the map
        if np.amin(c) < 0 or \
           c[0] >= grid.shape[0] or \
           c[1] >= grid.shape[1]:
            hit = True
            break
        # Next check if we're in collision
        if grid[c[0], c[1]] >= alt_offset:
            hit = True
            break
    return hit


def in_line(p1, p2, p3, epsilon, normalize=False):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    if normalize:
        p1 = p1 / LA.norm(p1)
        p2 = p2 / LA.norm(p2)
        p3 = p3 / LA.norm(p3)
    m = np.vstack((p1, p2, p3))
    if m.shape[-1] == 2:
        m = np.hstack((m, np.ones((3,1))))
    return np.abs(LA.det(m)) < epsilon


def prune_path(path, epsilon, normalize=False):
    pruned_path = []
    last, cand = path[:2]
    pruned_path.append(last)
    for p in path[2:]:
        if not in_line(last, cand, p, epsilon, normalize):
            pruned_path.append(cand)
            last = pruned_path[-1]
        cand = p
    if last != path[-1]:
        pruned_path.append(path[-1])
    return pruned_path
