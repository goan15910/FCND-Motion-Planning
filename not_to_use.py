import numpy as np

from planning_utils import Map_graph, Action, Action_3D, Poly, valid_actions, valid_cell


class Random_graph(Map_graph):
    """
    Create 3D random sampled graph.
    """
    def __init__(self, data, verbose=False):
        Map_graph.__init__(self, data, verbose=verbose)


    def NEAH_to_graph3d(self, path):
        new_path = []
        for p in path:
            new_p = np.array(p) - self.map_min
            new_path.append(tuple(new_p.astype(np.int64)))
        return new_path


    def NEAH_path(self, path):
        new_path = []
        for p in path:
            new_p = np.zeros(4, dtype=np.int64)
            new_p[:3] = p[:3]
            new_path.append(tuple(new_p.astype(np.int64)))
        return new_path


    def create_graph(self, start, goal):
        # setup start, goal
        self.set_start_goal(start, goal)

        # nodes and polygons
        self.extract_polygons()
        nodes = self.sample_nodes()
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))
        nodes.extend([start, goal])

        # create graph
        g = nx.Graph()
        tree = KDTree(nodes)
        for n1 in nodes:
            idxs = tree.query([n1], self.n_neighbors, return_distance=False)[0]

            for idx in idxs:
                n2 = nodes[idx]
                if n2 == n1:
                    continue

                if self.valid_edge(n1, n2):
                    dist = LA.norm(np.array(n1)-np.array(n2))
                    g.add_edge(n1, n2, weight=dist)

        if self.verbose:
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            print("Create graph with {} nodes, {} edges".format(n_nodes, n_edges))

        return g


    def next_nodes(self, n1, n1_cost):
        nodes = []
        for n2 in self.graph.neighbors(n1):
            a_cost = self.graph.edges[n1, n2]['weight']
            n2_cost = n1_cost + a_cost + self.h_func(n2, self.goal)
            nodes.append((tuple(n2), n2_cost))
        return nodes


    def extract_polygons(self):
        polys = []
        for i in range(self.data.shape[0]):
            n, e, a, d_n, d_e, d_a = self.data[i, :]
            corners = [
                (n - d_n, e - d_e),
                (n - d_n, e + d_e),
                (n + d_n, e + d_e),
                (n + d_n, e - d_e)]
            height = a + d_a
            p = Poly(corners, height)
            polys.append(p)
        self.polys = polys


    def sample_nodes(self):
        if self.verbose:
            print("sampling nodes ...")
        num = self.n_sample
        nvals = np.random.uniform(self.n_min, self.n_max, num)
        evals = np.random.uniform(self.e_min, self.e_max, num)
        avals = np.random.uniform(self.drone_alt, self.a_max, num)

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
                valid_nodes.append(tuple(map(int, n)))

        return valid_nodes


    def valid_edge(self, n1, n2):
        l = LineString([n1, n2])
        for p in self.polys:
            min_h = min(n1[2], n2[2])
            if p.crosses(l) and p.height >= min_h:
                return False
        return True