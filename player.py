from abc import ABC, abstractmethod
import random

import networkx as nx

from games.game import Game, ActType
from utils.game_utils import mask_illegal_actions
from utils.utils import dictkv_to_dictvk


class Player(ABC):
    """Player abstract class"""

    @abstractmethod
    def play(self, game: Game) -> ActType:
        """Select an action to play"""


class GreedyPlayer(Player):
    """Player that play various strategies of greedy graph coloring"""

    def __init__(self, strategy='largest_first') -> None:
        self.strategy = strategy
        self.colors = None
    
    def play(self, game: Game) -> ActType:
        if colors is None:
            self.colors = nx.greedy_color(game.graph, self.strategy)
            self.colors = dictkv_to_dictvk(self.colors)
        action = self.colors.pop(self.colors.keys()[0])


class HumanPlayer(Player):
    """Human player"""

    def play(self, game: Game) -> ActType:
        action = None
        while action not in mask_illegal_actions(game.legal_actions()):
            action = int(input(f'Enter your move: '))
        return action


class RandomPlayer(Player):
    """Player that plays randomly"""

    def play(self, game: Game) -> ActType:
        actions = mask_illegal_actions(game.legal_actions())
        return random.choices(legal_actions)
