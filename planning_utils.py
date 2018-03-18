from enum import Enum
from queue import PriorityQueue

import numpy as np
import numpy.linalg as LA

from shapely.geometry import Polygon, Point, LineString


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min + 1)))
    east_size = int(np.ceil((east_max - east_min + 1)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


class Poly:
    """Class help to maintain graph"""
    def __init__(self, coords, height):
        self._poly = Polygon(coords)
        self._h = height

    @property
    def height(self):
        return self._h

    @property
    def coords(self):
        return list(self._poly.exterior.coords[:-1])

    @property
    def area(self):
        return self._poly.area

    @property
    def center(self):
        return (self._poly.centroid.x, self._poly.centroid.y)

    def contains(self, point):
        point = Point(point)
        return self._poly.contains(point)

    def crosses(self, other):
        return self._poly.crosses(other)


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


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)
    if (x - 1 < 0 or y + 1 > m) or grid[x-1, y+1] == 1:
        valid_actions.remove(Action.NORTH_EAST)
    if (x + 1 > m or y + 1 > m) or grid[x+1, y+1] == 1:
        valid_actions.remove(Action.SOUTH_EAST)
    if (x + 1 > m or y - 1 < 0) or grid[x+1, y-1] == 1:
        valid_actions.remove(Action.SOUTH_WEST)
    if (x - 1 < 0 or y - 1 < 0) or grid[x-1, y-1] == 1:
        valid_actions.remove(Action.NORTH_WEST)

    return valid_actions


class A_star:
    """
    A star algorithm for 2D grid / 3D grid / graph
    """
    def __init__(self,
                 graph,
                 start,
                 goal,
                 actions=None,
                 h_func=None):
        # setup basic component of A*
        self.graph = graph
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.actions = actions
        if h_func is None:
            self.h_func = heuristic
        else:
            self.h_func = h_func


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
                print('Found a path.')
                found = True
                break
            else:
                n2_list = self.next_nodes(n1, n1_cost)
                for n2,n2_cost in n2_list:
                    if tuple(n2) not in visited:
                        visited.add(tuple(n2))
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


    def next_nodes(self, n1, n1_cost):
        nodes = []
        if self.actions is None:
            for n2 in self.graph[n1]:
                a_cost = self.graph.edges[n1, n2]['weight']
                n2_cost = n1_cost + a_cost + self.h_func(n2, self.goal)
                nodes.append((tuple(n2), n2_cost))
        else:
            for a in self._valid_actions(n1):
                mapped = map(lambda x,y:x+y, n1, a.delta)
                n2 = tuple(mapped)
                n2_cost = n1_cost + a.cost + self.h_func(n2, self.goal)
                nodes.append((n2, n2_cost))
        return nodes


    def _valid_actions(self, cur_node):
        assert self.actions is not None, \
            "Graph has no action set"
        valid_actions = list(self.actions)
        for a in valid_actions:
            mapped =  map(sum, zip(cur_node, a.delta))
            next_node = tuple(mapped)
            if not self._valid_grid_cell(next_node):
                valid_actions.remove(a)

        return valid_actions


    def _valid_grid_cell(self, node):
        shape = self.graph.shape
        n_dim = len(shape)
        node = tuple(map(int, node))
        for i in range(n_dim):
            if (node[i] < 0) or \
               (node[i] > shape[i]-1) or \
               (self.graph[node] == 1):
                return False
        return True


def a_star(grid, h, start, goal):
    """
    Given a grid and heuristic function returns
    the lowest cost path from start to goal.
    """
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        n1_cost = item[0]
        n1 = item[1]

        if n1 == goal:
            print('Found a path.')
            found = True
            break
        else:
            # Get the new vertexes connected to the current vertex
            for a in valid_actions(grid, n1):
                n2 = list(map(lambda x,y:x+y, n1, a.delta))
                n2_cost = n1_cost + a.cost + h(n2, goal)

                if n2 not in visited:
                    visited.add(n2)
                    queue.put((n2_cost, n2))
                    branch[n2] = (n2_cost, n1, a)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 

    return path[::-1], path_cost


def heuristic(position, goal_position):
    return LA.norm(np.array(position) - np.array(goal_position))


def in_line(p1, p2, p3, epsilon):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    m = np.vstack((p1, p2, p3))
    m = np.hstack((m, np.ones((3,1))))
    return LA.det(m) < epsilon


def prune_path(path, epsilon):
    pruned_path = []
    last, cand = path[:2]
    pruned_path.append(last)
    for p in path[2:]:
        if not in_line(last, cand, p, epsilon):
            pruned_path.append(cand)
            last = pruned_path[-1]
        cand = p
    if last != path[-1]:
        pruned_path.append(path[-1])
    return pruned_path
