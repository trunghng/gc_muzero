from abc import ABC, abstractmethod
import random

from game import Game, ActType


class Player(ABC):
    """Player abstract class"""

    @abstractmethod
    def play(self, game: Game) -> ActType:
        """Select an action to play"""


class RandomPlayer(Player):
    """Player that plays randomly"""

    def play(self, game: Game) -> ActType:
        legal_actions = game.legal_actions()
        return random.choices(legal_actions)[0]


class HumanPlayer(Player):
    """Human player"""

    def play(self, game: Game) -> ActType:
        action = None
        while action not in game.legal_actions():
            action_str = input(f'Enter your move: ')
            action = int(action_str)
        return action
