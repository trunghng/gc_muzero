from typing import Tuple, Dict, List, Any
import random

import ray
import torch
import numpy as np
from torch_geometric.data import Data, Batch

from games.game import GameHistory
from shared_storage import SharedStorage
from utils.utils import ftensor, set_seed


@ray.remote
class ReplayBuffer:

    def __init__(self,
                 initial_checkpoint: Dict[str, Any],
                 initial_buffer: List[GameHistory],
                 config) -> None:
        set_seed(config.seed)
        self.config = config
        self.memory = initial_buffer
        self.size = config.buffer_size
        self.played_games = initial_checkpoint['played_games']
        self.played_steps = initial_checkpoint['played_steps']
        self.reanalysed_games = initial_checkpoint['reanalysed_games']

    def len(self) -> int:
        return len(self.memory)

    def get_buffer(self) -> List[GameHistory]:
        return self.memory

    def add(self, game_history: GameHistory, shared_storage: SharedStorage) -> None:
        """Store history of a new game into the buffer"""
        # Remove the oldest game from the buffer if the max size is reached
        if len(self.memory) == self.size:
            self.memory.pop(0)
        self.memory.append(game_history)
        self.played_games += 1
        self.played_steps += len(game_history)

        shared_storage.set_info.remote({
            'played_games': self.played_games,
            'played_steps': self.played_steps
        })

    def sample_n_games(
        self, n: int
    ) -> Tuple[List[int], List[GameHistory]] | Tuple[int, GameHistory]:
        selected_indices = np.random.choice(range(len(self.memory)), size=n)
        game_histories = np.asarray(self.memory)[selected_indices]
        if n == 1:
            return selected_indices[0], game_histories[0]
        return selected_indices, game_histories

    def update_game(
        self,
        idx: int,
        game_history: GameHistory,
        shared_storage: SharedStorage
    ) -> None:
        self.memory[idx] = game_history
        self.reanalysed_games += 1
        shared_storage.set_info.remote({'reanalysed_games': self.reanalysed_games})

    def sample(
        self
    ) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return observation_batch:
        :return action_batch:           (B x (unroll_steps + 1))
        :return value_target_batch:     (B x (unroll_steps + 1))
        :return reward_target_batch:    (B x (unroll_steps + 1))
        :return policy_target_batch:    (B x (unroll_steps + 1) x n_actions)
        """
        _, game_histories = self.sample_n_games(self.config.batch_size)
        batch = [[], [], [], [], []]

        for game_history in game_histories:
            t = random.randint(0, len(game_history))

            observations = game_history.stack_n_observations(
                t,
                self.config.n_stacked_observations,
                self.config.n_actions,
                self.config.stack_action
            )

            encoded_actions = game_history.encoded_actions[t:t + self.config.unroll_steps + 1]
            if len(encoded_actions) < self.config.unroll_steps + 1:
                absorbed_indices = np.random.choice(
                    range(len(game_history.encoded_actions)),
                    size=self.config.unroll_steps + 1 - len(encoded_actions)
                )
                encoded_actions.extend([game_history.encoded_actions[i] for i in absorbed_indices])

            value_targets, reward_targets, policy_targets = game_history.make_target(
                t,
                self.config.td_steps,
                self.config.gamma,
                self.config.unroll_steps,
                self.config.n_actions
            )
            batch[0].append(observations)
            batch[1].append(encoded_actions)
            batch[2].append(value_targets)
            batch[3].append(reward_targets)
            batch[4].append(policy_targets)

        batch[0] = Batch.from_data_list(batch[0])
        for i in range(1, len(batch)):
            batch[i] = ftensor(np.asarray(batch[i]))
        return tuple(batch)
