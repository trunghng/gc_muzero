import random
import math
from typing import List, Tuple
from copy import deepcopy
import os
import pickle

import networkx as nx


def save_dataset(save_dir: str, dataset: List[nx.Graph]) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle.dump(dataset, open(os.path.join(save_dir, 'dataset.pkl'), 'wb'))


def load_dataset(dataset_path: str) -> List[nx.Graph]:
    return pickle.load(open(dataset_path, 'rb'))


def generate_complete_graph(n: int) -> nx.Graph:
    return nx.complete_graph(n)


def generate_graphs(n: int, types: List[str], n_graphs: int, k: int) -> List[nx.Graph]:
    graphs = []
    if 'ER' in types:
        graphs.extend([generate_erdos_renyi_graph(n) for _ in range(n_graphs)])
    if 'BA' in types:
        graphs.extend([generate_barabasi_albert_graph(n) for _ in range(n_graphs)])
    if 'WS' in types:
        graphs.extend([generate_watts_strogatz_graph(n) for _ in range(n_graphs)])
    if 'LT' in types:
        assert k is not None, "The chromatic number should be provided to generate Leighton graph"
        graphs.extend([generate_leighton_graph(n, k) for _ in range(n_graphs)])

    return graphs


def generate_erdos_renyi_graph(n: int, p: float=None) -> nx.Graph:
    if p is None:
        p = random.uniform(0.01, 0.99)
    graph = None
    while graph is None or not graph.edges:
        graph = nx.erdos_renyi_graph(n, p)
    return graph


def generate_barabasi_albert_graph(n: int, m: int=None) -> nx.Graph:
    pass


def generate_watts_strogatz_graph(n: int, k: int=None, p: float=None) -> nx.Graph:
    pass


def generate_leighton_graph(n: int, k: int):
    return LeightonGraph(n, k).generate()


class LeightonGraph:
    """
    n-vertex graph with chromatic number k generator using Leighton's algorithm (assuming k|n)
    [1] Leighton, Frank Thomson. A Graph Coloring Algorithm for Large Scheduling Problems.
        Journal of research of the National Bureau of Standards 84 6 (1979): 489-506.

    :param n: number of nodes
    :param k: chromatic number
    """

    PRIME_TO_1000 = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
        83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
        173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
        269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367,
        373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
        467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587,
        593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683,
        691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
        821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929,
        937, 941, 947, 953, 967, 971, 977, 983, 991, 997
    ]

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k
        self.n_prime_factors = self._prime_factorization(self.n)
        self.k_prime_factors = self._prime_factorization(self.k)

    def _prime_factorization(self, x: int) -> List[int]:
        """Get the unique prime factorization of x"""
        primes_factors = []
        
        while x % 2 == 0:
            x //= 2
            primes_factors.append(2)

        p = 3
        while p <= math.sqrt(x):
            if x % p == 0:
                x //= p
                primes_factors.append(p)
            else:
                p += 2

        # x is a prime itself
        if x > 1:
            primes_factors.append(x)

        return primes_factors

    def _generate_m(self) -> Tuple[int, List[int]]:
        """
        Generate m and its prime decomposition s.t.
        (i)  m >> n
        (ii) gcd(n, m) = k
        """
        min_m = self.k * (self.n ** 2)

        n_prime_factors_notin_k = deepcopy(self.n_prime_factors)
        for p in self.k_prime_factors:
            n_prime_factors_notin_k.remove(p)

        remaining_primes = set(self.PRIME_TO_1000) - set(n_prime_factors_notin_k)
        m = self.k
        m_prime_factors = deepcopy(self.k_prime_factors)
        while m < min_m:
            random_p = random.choice(tuple(remaining_primes))
            m *= random_p
            m_prime_factors.append(random_p)

        return m, m_prime_factors

    def _generate_c(self, m: int, m_prime_factors: List[int]) -> int:
        """
        Generate c s.t.
            gcd(c, m) = 1
        """
        min_c = 10 * self.k * self.n
        remaining_primes = set(self.PRIME_TO_1000) - set(m_prime_factors)
        c = 1

        while c < min_c:
            random_p = random.choice(tuple(remaining_primes))
            c *= random_p
        return c

    def _generate_a(self, m: int, m_prime_factors: List[int]) -> int:
        """
        Generate a s.t.
        (i)  p|m -> p|a - 1 for all p
        (ii) 4|m -> 4|a - 1
        """
        min_a = 10 * self.k * self.n
        a_minus_1 = 4 if m % 4 == 0 else 1

        for p in set(m_prime_factors):
            a_minus_1 *= p

        while a_minus_1 < min_a:
            random_p = random.choice(self.PRIME_TO_1000)
            a_minus_1 *= random_p

        return a_minus_1 + 1

    def _generate_b(self) -> List[int]:
        """
        Generate a list b = [b_j, b_{j-1}, ..., b_2] where b_i is the number of i-cliques inside the graph
        """
        return [random.randint(1, int(self.n / self.k) + 1)] + \
            [random.randint(0, int(self.n / (self.k - i) + 1)) for i in range(1, self.k - 1)]

    def _generate_parameters(self) -> Tuple[int, int, int, List[int]]:
        m, m_prime_factors = self._generate_m()
        c = self._generate_c(m, m_prime_factors)
        a = self._generate_a(m, m_prime_factors)
        b = self._generate_b()
        return m, c, a, b

    def _generate_x(self, m: int, c: int, a: int, b: List[int]) -> List[int]:
        """
        Generate a sequence of integers {x_i} in [0, m-1] s.t.
        (i)  There is no duplication in every m consecutive elements (i.e. x_i, ..., x_{i+m-1})
        (ii) x_i = x_{i+m}

        Staring from an initial x_0 in [0, m-1] and using the update rule
            x_i = (a * x_{i-1} + c) mod m
        """
        x_0 = random.randint(0, m)
        x = [x_0]
        clique_sizes = [len(b) - i + 1 for i in range(len(b))]
        x_len = sum([b_i * clique_size for b_i, clique_size in zip(b, clique_sizes)]) + 1

        for i in range(1, x_len):
            x.append((a * x[i - 1] + c) % m)
        return x

    def _generate_y(self, x: List[int]) -> List[int]:
        """
        Generate a sequence of integers {y_i} in [0, n-1] from {x_i}:
            y_i = x_i mod n
        """
        return [x_i % self.n for x_i in x]

    def _generate_clique_edges(self, clique_vertices: List[int]) -> List[Tuple[int, int]]:
        """Generate edges of a clique from its vertices"""
        clique_edges = []
        for i, v_i in enumerate(clique_vertices):
            for v_j in clique_vertices[i + 1:]:
                clique_edges.append((min(v_i, v_j), max(v_i, v_j)))
        return clique_edges

    def _generate_edges(self, m: int, c: int, a: int, b: List[int]) -> List[Tuple[int, int]]:
        """
        Generate the graph edges by following the procedure (note that b = [b_k, ..., b_2])
        (1) Select the first k values of {y_i} beginning with y_1 and add the corresponding edges to E.
        (2) If b_k > 1, select the next k values of {y_i} and add the corresponding edges to E.
        (3) Repeat (1), (2) until b_k k-cliques have been implanted in G.
        (4) Add, in an identical fashion, b_{k-1} (k-1)-cliques to G.
        (5) Continue the process until b_2 2-cliques (edges) have been added to E.
        """
        x = self._generate_x(m, c, a, b)
        y = self._generate_y(x)

        E = []
        y_idx = 1
        for i, bi in enumerate(b):
            clique_size = self.k - i
            for _ in range(bi):
                clique_vertices = y[y_idx: y_idx + clique_size]
                clique_edges = self._generate_clique_edges(clique_vertices)
                E += clique_edges
                y_idx += clique_size

        return set(E)

    def generate(self) -> nx.Graph:
        graph = nx.Graph()

        while len(graph.nodes) < self.n:
            m, c, a, b = self._generate_parameters()
            edges = self._generate_edges(m, c, a, b)
            graph.add_edges_from(edges)

        graph_ = nx.Graph()
        graph_.add_nodes_from(sorted(graph.nodes))
        graph_.add_edges_from(graph.edges)
        return graph_
