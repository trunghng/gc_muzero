import random

import numpy as np
import torch
import networkx as nx

import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_erdos_renyi_graph(n: int, p: float=None):
    if p is None:
        p = random.uniform(0.1, 0.9)
    return nx.erdos_renyi_graph(n, p)