from copy import deepcopy
from typing import Dict, Any

import ray
import numpy as np

from game import Game, GameHistory
from mcts import MCTS
from network import MuZeroNetwork
from utils import set_seed
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer


@ray.remote
class SelfPlay:

    def __init__(self,
                 game: Game,
                 initial_checkpoint: Dict[str, Any],
                 config,
                 seed: int) -> None:
        set_seed(seed)
        self.config = config
        self.game = game
        self.mcts = MCTS(self.config)
        self.network = MuZeroNetwork(config.observation_dim,
                                     config.action_space_size,
                                     config.embedding_size,
                                     config.dynamics_layers,
                                     config.reward_layers,
                                     config.policy_layers,
                                     config.value_layers,
                                     config.support_limit)
        self.network.set_weights(initial_checkpoint['model_state_dict'])
        self.network.eval()

    def play_continuously(self,
                          shared_storage: SharedStorage,
                          replay_buffer: ReplayBuffer,
                          test: bool=False) -> None:
        while ray.get(shared_storage.get_info.remote('training_step')) < self.config.training_steps:
            self.network.set_weights(ray.get(shared_storage.get_info.remote('model_state_dict')))
            if test:
                game_history = self.play(0) # select action with max #visits
                shared_storage.set_info.remote({
                    'episode_length': len(game_history),
                    'episode_return': game_history.compute_return(self.config.gamma),
                    'mean_value': np.mean([v for v in game_history.root_values if v])
                })
            else:
                game_history = self.play(
                    self.config.visit_softmax_temperature_func(
                        self.config.training_steps, 
                        ray.get(shared_storage.get_info.remote('training_step'))
                    )
                )
                replay_buffer.add.remote(game_history, shared_storage)

    def play(self, temperature: float, render: bool=False) -> GameHistory:
        """Run a self-play game"""
        observation, colors = self.game.reset()
        game_history = GameHistory(self.game)

        while True:
            stacked_observations = game_history.stack_n_observations(
                -1,
                self.config.stacked_observations,
                self.config.action_space_size,
                self.config.stack_action
            )
            root = self.mcts.search(
                self.network,
                stacked_observations,
                self.game.legal_actions(),
                game_history.actions,
                self.game.action_encoder
            )
            action = self.mcts.select_action(root, temperature)
            action_probs = self.mcts.action_probabilities(root)
            next_observation, reward, terminated = self.game.step(action)
            game_history.save(observation, action, reward, action_probs, root.value(), colors)
            observation = next_observation
            colors = deepcopy(self.game.colors)
            if render:
                    self.game.render()

            if terminated or len(game_history) > self.config.max_moves:
                break
        return game_history
