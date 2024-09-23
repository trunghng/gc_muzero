from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple, TypeVar

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from utils.graph_utils import generate_complete_graph

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


class Game(ABC):
    """Game abstract class"""

    def __init__(self, observation_dim: List[int], n_actions: int) -> None:
        self.observation_dim = observation_dim
        self.n_actions = n_actions

    @abstractmethod
    def reset(self) -> ObsType:
        """"""

    @abstractmethod
    def terminated(self) -> bool:
        """"""

    @abstractmethod
    def legal_actions(self) -> List[ActType]:
        """"""

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        """"""

    @abstractmethod
    def observation(self) -> ObsType:
        """"""

    @abstractmethod
    def action_encoder(self, action: ActType) -> ActType:
        """"""

    @abstractmethod
    def visit_softmax_temperature_func(self,
                                       training_steps: int,
                                       training_step: int) -> float:
        """
        :param training_steps: number of training steps
        :param training_step: current training step
        """

    @abstractmethod
    def render(self) -> None:
        """"""


class GameHistory:
    """
    For atari games, an action does not necessarily have a visible effect on 
    the observation, we encode historical actions into the stacked observation.
    """

    def __init__(self, game: Game) -> None:
        self.observations = []          # o_t: State observations
        self.actions = []               # a_{t+1}: Action leading to transition s_t -> s_{t+1}
        self.encoded_actions = []
        self.rewards = []               # u_{t+1}: Observed reward after performing a_{t+1}
        self.action_probabilities = []  # pi_t: Action probabilities produced by MCTS
        self.root_values = []           # v_t: MCTS value estimation
        self.colors = []                # color list at time-step t, used in Reanalyse
        self.reanalysed_action_probabilities = None
        self.reanalysed_root_values = None
        self.action_encoder = game.action_encoder
        self.initial_observation = Data(
            x=torch.tensor(game.observation(), dtype=torch.float32, requires_grad=False),
            edge_index=torch.tensor(
                list(game.graph.edges), dtype=torch.int64, requires_grad=False
            ).transpose(0, 1),
            edge_attr=torch.tensor(game.edge_features, dtype=torch.int64, requires_grad=False)
            if game.complete_graph else None
        )

    def __len__(self) -> int:
        return len(self.observations)

    def save(self,
             observation: ObsType,
             action: ActType,
             reward: float,
             pi: List[float],
             root_value: float,
             colors: List[int]) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.encoded_actions.append(self.action_encoder(action))
        self.rewards.append(reward)
        self.action_probabilities.append(pi)
        self.root_values.append(root_value)
        self.colors.append(colors)

    def save_reanalysed_stats(self,
                              action_probabilities: List[List[float]],
                              root_values: List[float]) -> None:
        self.reanalysed_action_probabilities = action_probabilities
        self.reanalysed_root_values = root_values

    def stack_n_observations(self,
                             t: int,
                             n: int,
                             n_actions: int,
                             stack_action: bool=False) -> Data:
        """
        Stack n most recent observations (and corresponding 
        actions lead to the states with Atari) upto 't':
            o_{t-n+1}, ..., o_t

        :param t: time step of the latest observation to stack
        :param n: number of observations to stack
        :param n_actions: size of the action space
        :param stack_action: whether to stack historical actions
        """
        planes = []

        if len(self) == 0:
            planes.append(self.initial_observation.x)
            if stack_action:
                planes.append(np.zeros_like(self.initial_observation.x))
            for _ in range(n - 1):
                planes.append(np.zeros_like(self.initial_observation.x))
                if stack_action:
                    planes.append(np.zeros_like(self.initial_observation.x))
        else:
            # Convert to positive index
            t = t % len(self)
            n_ = min(n, t + 1)

            for step in reversed(range(t - n_ + 1, t + 1)):
                planes.append(self.observations[step])
                if stack_action:
                    planes.append(np.full_like(self.observations[step], self.actions[step] / n_actions))

            # If n_stack_observations > t + 1, we attach planes of zeros instead
            for _ in range(n - n_):
                planes.append(np.zeros_like(self.observations[step]))
                if stack_action:
                    planes.append(np.zeros_like(self.observations[step]))

        n = np.concatenate(planes, axis=0)
        return Data(
            x=torch.tensor(n, dtype=torch.float32, requires_grad=False),
            edge_index=self.initial_observation.edge_index,
            edge_attr=self.initial_observation.edge_attr
        )

    def compute_return(self, gamma: float) -> float:
        """
        Compute episode return, assuming that the game is over
        G = r_1 + r_2 + ... + r_T

        :param gamma: discount factor
        """
        eps_return = sum(self.rewards)
        return eps_return

    def make_target(
        self,
        t: int,
        td_steps: int,
        gamma: float,
        unroll_steps: int,
        n_actions: int
    ) -> Tuple[List[float], List[float], List[List[float]]]:
        """
        Create targets for every unroll steps

        :param t: current time step
        :param td_steps: n-step TD
        :param gamma: discount factor
        :param unroll_steps: number of unroll steps
        :param n_actions: size of the action space
        :return: value targets, reward targets, policy targets
        """
        value_targets, reward_targets, policy_targets = [], [], []

        def _compute_value_target(step: int) -> float:
            """
            Compute value target
            - For board games, value target is the total reward from the current index til the end
            - For other games, value target is the discounted root value of the search tree
            'td_steps' into the future, plus the discounted sum of all rewards until then

            z_t = u_{t+1} + gamma * u_{t+2} + ... + gamma^{n-1} * u_{t+n} + gamma^n * v_{t+n} 
            """
            if gamma == 1:
                rewards = [self.rewards[i] for i in range(step, len(self))]
                value = sum(rewards)
            else:
                bootstrap_step = step + td_steps
                if bootstrap_step < len(self):
                    bootstrap = self.reanalysed_root_values[bootstrap_step]\
                        if self.reanalysed_root_values else self.root_values[bootstrap_step]
                    bootstrap *= gamma ** td_steps
                else:
                    bootstrap = 0

                discounted_rewards = [
                    reward * gamma ** k
                    for k, reward in enumerate(self.rewards[step + 1: bootstrap_step + 1])
                ]
                value = sum(discounted_rewards) + bootstrap
            return value

        for step in range(t, t + unroll_steps + 1):
            value = _compute_value_target(step)

            if step < len(self):
                value_targets.append(value)
                reward_targets.append(self.rewards[step])
                policy_targets.append(
                    self.reanalysed_action_probabilities[step] if self.reanalysed_action_probabilities
                    else self.action_probabilities[step]
                )
            else:
                value_targets.append(0)
                reward_targets.append(0)
                policy_targets.append([1 / n_actions] * n_actions)

        return value_targets, reward_targets, policy_targets

class GraphColoring(Game):

    def __init__(self, graphs: List[nx.Graph], complete_graph: bool=False) -> None:
        self.graphs = graphs
        self.complete_graph = complete_graph
        super().__init__([len(self.graphs[0].nodes), 2], len(self.graphs[0].nodes))

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
        """Each non-colored node represents a potential action"""
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

    def observation(self) -> ObsType:
        """Using feature matrix as the observation, each feature vector is a tuple of the node label and color of that node"""
        feature_matrix = [[len(list(self.graph.neighbors(d))), c] for (d, c) in zip(self.graph.nodes, self.colors)]
        return feature_matrix

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
