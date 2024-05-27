from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from utils import generate_complete_graph

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
PlayerType = TypeVar('PlayerType')


class Game(ABC):
    """Game abstract class"""

    def __init__(self,
                players: int,
                observation_dim: List[int],
                action_space: List[ActType]) -> None:
        self.players = players
        self.observation_dim = observation_dim
        self.action_space = action_space
        self.to_play = -1 if players == 2 else 0


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
    def visit_softmax_temperature_func(self, training_steps: int, training_step: int) -> float:
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
        self.to_plays = []              # p_t: Current player
        self.action_probabilities = []  # pi_t: Action probabilities produced by MCTS
        self.root_values = []           # v_t: MCTS value estimation
        self.action_encoder = game.action_encoder
        self.initial_observation = Data(
            x=torch.tensor(game.observation(), dtype=torch.float32, requires_grad=False),
            edge_index=torch.tensor(list(game.graph.edges), dtype=torch.int64, requires_grad=False).transpose(0, 1),
            edge_attr=torch.tensor(game.edge_features, dtype=torch.int64, requires_grad=False) if game.complete_graph else None
        )


    def __len__(self) -> int:
        return len(self.observations)


    def save(self,
            observation: ObsType,
            action: ActType,
            reward: float,
            to_play: PlayerType,
            pi: List[float],
            root_value: float) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.encoded_actions.append(self.action_encoder(action))
        self.rewards.append(reward)
        self.to_plays.append(to_play)
        self.action_probabilities.append(pi)
        self.root_values.append(root_value)


    def stack_observations(self,
                            t: int,
                            stacked_observations: int,
                            action_space_size: int,
                            stack_action: bool=False) -> np.ndarray:
        """
        Stack 'stacked_observations' most recent observations (and corresponding 
        actions lead to the states with Atari) upto 't':
            o_{t-stacked_observations+1}, ..., o_t

        :param t: time step of the latest observation to stack
        :param stacked_observations: number of observations to stack
        :param action_space_size: size of the action space
        :param stack_action: whether to stack historical actions
        """
        planes = []

        if len(self) == 0:
            planes.append(self.initial_observation.x)
            if stack_action:
                planes.append(np.zeros_like(self.initial_observation.x))
            for _ in range(stacked_observations - 1):
                planes.append(np.zeros_like(self.initial_observation.x))
                if stack_action:
                    planes.append(np.zeros_like(self.initial_observation.x))
        else:
            # Convert to positive index
            t = t % len(self)
            stacked_observations_ = min(stacked_observations, t + 1)

            for step in reversed(range(t - stacked_observations_ + 1, t + 1)):
                planes.append(self.observations[step])
                if stack_action:
                    planes.append(np.full_like(self.observations[step], self.actions[step] / action_space_size))

            # If n_stack_observations > t + 1, we attach planes of zeros instead
            for _ in range(stacked_observations - stacked_observations_):
                planes.append(np.zeros_like(self.observations[step]))
                if stack_action:
                    planes.append(np.zeros_like(self.observations[step]))

        stacked_observations = np.concatenate(planes, axis=0)
        return Data(
            x=torch.tensor(stacked_observations, dtype=torch.float32, requires_grad=False),
            edge_index=self.initial_observation.edge_index,
            edge_attr=self.initial_observation.edge_attr
        )


    def compute_return(self, gamma: float) -> float:
        """
        Compute episode return, assuming that the game is over
        G = r_1 + gamma * r_2 + ... + gamma^{T-1} * r_T

        :param gamma: discount factor
        """
        eps_return = self.rewards[-1]
        for r in reversed(self.rewards[0:-1]):
            eps_return = eps_return * gamma + r
        return eps_return


    def make_target(self,
                    t: int,
                    td_steps: int,
                    gamma: float,
                    unroll_steps: int,
                    action_space_size: int) -> Tuple[List[float], List[float], List[List[float]]]:
        """
        Create targets for every unroll steps

        :param t: current time step
        :param td_steps: n-step TD
        :param gamma: discount factor
        :param unroll_steps: number of unroll steps
        :param action_space_size: size of the action space
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
                rewards = []
                for i in range(step, len(self)):
                    rewards.append(self.rewards[i] if self.to_plays[i] == self.to_plays[step] else -self.rewards[i])
                value = sum(rewards)
            else:
                bootstrap_step = step + td_steps
                if bootstrap_step < len(self):
                    bootstrap = self.root_values[bootstrap_step] if self.to_plays[bootstrap_step] == self.to_plays[step]\
                                    else -self.root_values[bootstrap_step]
                    bootstrap *= gamma ** td_steps
                else:
                    bootstrap = 0

                discounted_rewards = [
                    (self.rewards[k] if self.to_plays[step + k] == self.to_plays[step] else -reward) * gamma ** k
                    for k in range(step + 1, bootstrap_step + 1)
                ]
                value = sum(discounted_rewards) + bootstrap
            return value

        for step in range(t, t + unroll_steps + 1):
            value = _compute_value_target(step)

            if step < len(self):
                value_targets.append(value)
                reward_targets.append(self.rewards[step])
                policy_targets.append(self.action_probabilities[step])
            else:
                value_targets.append(0)
                reward_targets.append(0)
                policy_targets.append([1 / action_space_size] * action_space_size)

        return value_targets, reward_targets, policy_targets


class GraphColoring(Game):

    def __init__(self, graphs: List[nx.Graph], complete_graph: bool) -> None:
        self.graphs = graphs
        self.complete_graph = complete_graph
        super().__init__(1, [len(self.graphs[0].nodes), 2], list(self.graphs[0].nodes))


    def reset(self) -> ObsType:
        idx = np.random.randint(len(self.graphs))
        if self.complete_graph:
            original_graph = self.graphs[idx]
            self.graph = generate_complete_graph(len(original_graph.nodes))
            self.edge_features = [1 if e in original_graph.edges else 0 for e in self.graph.edges]
        else:
            self.graph = self.graphs[idx]
        self.colors = [-1 for _ in range(len(self.graph.nodes))]
        return self.observation()


    def terminated(self) -> bool:
        """Game over when every nodes is colored"""
        return -1 not in self.colors


    def legal_actions(self) -> List[ActType]:
        """Each non-colored node represents a potential action"""
        return [i for i, c in enumerate(self.colors) if c == -1]


    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        """Taking a node as the input action"""
        self.select_color(action)
        terminated = self.terminated()
        reward = -(len(set(self.colors)) - 1) if terminated else 0
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
        feature_matrix = [[d, c] for (d, c) in zip(self.graph.nodes, self.colors)]
        return feature_matrix


    def action_encoder(self, action: ActType) -> ActType:
        """Encode action into one-hot style"""
        one_hot_action = [0 for _ in range(len(self.graph.nodes))]
        one_hot_action[action] = 1
        return one_hot_action


    def visit_softmax_temperature_func(self, training_steps: int, training_step: int) -> float:
        """
        :param training_steps: number of training steps
        :param training_step: current training step
        """


    def render(self) -> None:
        plt.figure(figsize=(7, 7))
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(self.graph, pos=nx.spring_layout(self.graph, seed=42), node_color=self.colors, cmap='tab20c')
        plt.show()
