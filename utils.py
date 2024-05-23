import random
from typing import List

import numpy as np
import torch
import networkx as nx

import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_graphs(n: int, types: List[str]=['ER'], n_graphs: int=1000) -> List[nx.Graph]:
    graphs = []
    if 'ER' in types:
        graphs.extend([generate_erdos_renyi_graph(n) for _ in range(n_graphs)])
    if 'BA' in types:
        graphs.extend([generate_barabasi_albert_graph(n) for _ in range(n_graphs)])
    if 'WS' in types:
        graphs.extend([generate_watts_strogatz_graph(n) for _ in range(n_graphs)])
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
