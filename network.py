import math
from typing import Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

from network_utils import mlp, support_to_scalar,\
    scalar_to_support, normalize_hidden_state


class RepresentationNetwork(nn.Module):
    """Representation network"""

    def __init__(self,
                 observation_dim: List[int],
                 embedding_size: int) -> None:
        super().__init__()
        self.nodes = observation_dim[0]
        self.conv1 = gnn.GraphConv(observation_dim[1], embedding_size)
        self.conv2 = gnn.GraphConv(embedding_size, embedding_size)
        self.conv3 = gnn.GraphConv(embedding_size, embedding_size)

    def forward(self, data: Data) -> torch.Tensor:
        """
        :return hidden_state:   (B x nodes x embedding_size)
        """
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = h.view(-1, self.nodes, h.shape[1])
        return h

class DynamicsNetwork(nn.Module):
    """Dynamics network"""

    def __init__(self,
                 observation_dim: List[int],
                 embedding_size: int,
                 dynamics_layers: List[int],
                 reward_layers: List[int],
                 support_size: int) -> None:
        super().__init__()
        self.nodes = observation_dim[0]
        self.hidden_state_network = mlp([
            self.nodes * (embedding_size + 1),
            *dynamics_layers,
            self.nodes * embedding_size
        ])
        self.reward_network = mlp([
            self.nodes * embedding_size, *reward_layers, support_size
        ])

    def forward(self, state_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param state_action:        (B x nodes x (embedding_size + 1))
        :return next_hidden_state:  (B x nodes x embedding_size)
        :return reward_logits:      (B x support_size)
        """
        next_hidden_state = self.hidden_state_network(state_action)
        reward_logits = self.reward_network(next_hidden_state)
        return next_hidden_state.view(
            next_hidden_state.shape[0], self.nodes, -1
        ), reward_logits

class PredictionNetwork(nn.Module):
    """Prediction network"""

    def __init__(self,
                 observation_dim: List[int],
                 action_space_size: int,
                 embedding_size: int,
                 policy_layers: List[int],
                 value_layers: List[int],
                 support_size: int) -> None:
        super().__init__()
        self.policy_network = mlp([
            observation_dim[0] * embedding_size, *policy_layers, action_space_size
        ])
        self.value_network = mlp([
            observation_dim[0] * embedding_size, *value_layers, support_size
        ])

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param hidden_state:    (B x nodes x embedding_size)
        :return policy_logits:  (B x action_space_size)
        :return value_logits:   (B x support_size)
        """
        policy_logits = self.policy_network(hidden_state)
        value_logits = self.value_network(hidden_state)
        return policy_logits, value_logits


class MuZeroNetwork(nn.Module):

    def __init__(self,
                 observation_dim: List[int],
                 action_space_size: int,
                 embedding_size: int,
                 dynamics_layers: List[int],
                 reward_layers: List[int],
                 policy_layers: List[int],
                 value_layers: List[int],
                 support_limit: int) -> None:
        super().__init__()
        self.support_limit = support_limit
        support_size = self.support_limit * 2 + 1
        
        self.repretation_network = RepresentationNetwork(
            observation_dim, embedding_size
        )
        self.dynamics_network = DynamicsNetwork(
            observation_dim, embedding_size, dynamics_layers,
            reward_layers, support_size
        )
        self.prediction_network = PredictionNetwork(
            observation_dim, action_space_size, embedding_size,
            policy_layers, value_layers, support_size
        )

    def representation(self, observation: Data) -> torch.Tensor:
        """
        :param observation: 
        :return hidden_state:   (B x nodes x embedding_size)
        """
        hidden_state = self.repretation_network(observation)
        return normalize_hidden_state(hidden_state)

    def dynamics(self,
                 hidden_state: torch.Tensor,
                 encoded_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param hidden_state:        (B x nodes x embedding_size)
        :param encoded_action:      (B x nodes x 1)
        :return next_hidden_state:  (B x nodes x embedding_size)
        :return reward_logits:      (B x support_size)
        """
        state_action = torch.cat((hidden_state, encoded_action), dim=-1)
        next_hidden_state, reward_logits = self.dynamics_network(
            state_action.view(state_action.shape[0], -1)
        )
        return normalize_hidden_state(next_hidden_state), reward_logits

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param hidden_state:    (B x nodes x embedding_size)
        :return policy_logits:  (B x action_space_size)
        :return value_logits:   (B x support_size)
        """
        policy_logits, value_logits = self.prediction_network(
            hidden_state.view(hidden_state.shape[0], -1)
        )
        return policy_logits, value_logits

    def initial_inference(
        self,
        observation: Data,
        inv_transform: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Representation + Prediction function

        :param observation:
        :param inv_transform: whether to apply inverse transformation on categorical form of value
        :return policy_logits:  (B x action_space_size)
        :return hidden_state:   (B x nodes x embedding_size)
        :return value:          (B x support_size) | (B)
        """
        hidden_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(hidden_state)
        if inv_transform:
            value = support_to_scalar(value_logits, self.support_limit)
            return policy_logits, hidden_state, value
        return policy_logits, hidden_state, value_logits

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        encoded_action: torch.Tensor,
        scalar_transform: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamics + Prediction function

        :param hidden_state:        (B x nodes x embedding_size)
        :param encoded_action:      (B x noes x 1)
        :param scalar_transform: whether to apply transformation on value and reward
        :return policy_logits:      (B x action_space_size)
        :return next_hidden_state:  (B x nodes x embedding_size)
        :return value:              (B x support_size) | (B)
        :return reward:             (B x support_size) | (B)
        """
        next_hidden_state, reward_logits = self.dynamics(hidden_state, encoded_action)
        policy_logits, value_logits = self.prediction(hidden_state)
        if scalar_transform:
            reward = support_to_scalar(reward_logits, self.support_limit)
            value = support_to_scalar(value_logits, self.support_limit)
            return policy_logits, next_hidden_state, value, reward
        return policy_logits, next_hidden_state, value_logits, reward_logits

    def recurrent_inference(self,
                            hidden_state: torch.Tensor,
                            encoded_action: torch.Tensor,
                            scalar_transform: bool=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamics + Prediction function

        :param hidden_state: (B x nodes x embedding_size)
        :param encoded_action: (B x nodes x 1)
        :param scalar_transform: whether to apply transformation on value and reward
        :return policy_logits: (B x action_space_size)
        :return next_hidden_state: (B x nodes x embedding_size)
        :return value: (1) |
        :return reward: (1) |
        """
        next_hidden_state, reward = self.dynamics(hidden_state, encoded_action)
        policy_logits, value = self.prediction(hidden_state)
        if scalar_transform:
            reward = support_to_scalar(reward, self.support_limit)
            value = support_to_scalar(value, self.support_limit)
        return policy_logits, next_hidden_state, value, reward

    def set_weights(self, weights: Any) -> None:
        if weights is not None:
            self.load_state_dict(weights)
