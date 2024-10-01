from copy import deepcopy
from typing import List, Tuple

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


from games.game import Game, ActType, ObsType
from utils.graph_utils import generate_complete_graph


class GraphColoring(Game):

    def __init__(self, graphs: List[nx.Graph], complete_graph: bool=False) -> None:
        self.graphs = graphs
        self.complete_graph = complete_graph
        super().__init__(
            [len(self.graphs[0].nodes), 2], len(self.graphs[0].nodes)
        )

    def reset(self) -> Tuple[ObsType, List[int]]:
        idx = np.random.randint(len(self.graphs))
        if self.complete_graph:
            original_graph = self.graphs[idx]
            self.graph = generate_complete_graph(len(original_graph.nodes))
            self.edge_features = [1 if e in original_graph.edges else 0 for e in self.graph.edges]
        else:
            self.graph = self.graphs[idx]
        self.colors = [-1 for _ in range(len(self.graph.nodes))]
        return self.observation(), deepcopy(self.colors)

    def terminated(self) -> bool:
        """Game over when every nodes is colored"""
        return -1 not in self.colors

    def legal_actions(self, colors: List[int] = None) -> np.ndarray:
        """
        Each non-colored node represents a potential action,
        which is denoted as 1, and 0 otherwise
        """
        if colors is None:
            colors = self.colors
        return np.where(np.asarray(colors) == -1, 1, 0)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        """Taking a node as the input action"""
        self.select_color(action)
        terminated = self.terminated()
        reward = -len(set(self.colors)) if terminated else 0
        return self.observation(), reward, terminated

    def select_color(self, node: int) -> None:
        current_colors = set(self.colors)
        neighbor_colors = [self.colors[n] for n in self.graph.neighbors(node)]
        
        selected_color = None
        for c in current_colors:
            if c not in neighbor_colors and c != -1:
                selected_color = c
                break
        if selected_color is None:
            selected_color = max(current_colors) + 1
        self.colors[node] = selected_color

    def feature_matrix(self) -> np.ndarray:
        # Each feature vector is a tuple of the node label and color of that node
        # feature_matrix = np.array([[len(list(self.graph.neighbors(d))), c] for (d, c) in zip(self.graph.nodes, self.colors)])
        feature_matrix = np.array([[np.sin(d), np.cos(d)] for _, d in self.graph.degree()])
        return feature_matrix

    def observation(self) -> ObsType:
        """Using feature matrix as the observation"""
        return self.feature_matrix()

    def action_encoder(self, action: ActType) -> ActType:
        """Encode action into one-hot style"""
        one_hot_action = [0 for _ in range(self.n_actions)]
        one_hot_action[action] = 1
        return one_hot_action

    def visit_softmax_temperature_func(self, training_steps: int, training_step: int) -> float:
        """
        :param training_steps: number of training steps
        :param training_step: current training step
        """
        return 1.0

    def render(self) -> None:
        plt.figure(figsize=(7, 7))
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(self.graph, pos=nx.spring_layout(self.graph, seed=42), node_color=self.colors, cmap='tab20c')
        plt.show()
