import argparse

import torch

from game import Game, GraphColoring
from arena import Arena
from player import Player, RandomPlayer, HumanPlayer, MuZeroPlayer
from muzero import MuZero
from utils import generate_graphs


def create_player(player: str) -> Player:
    if player == 'random':
        return RandomPlayer()
    elif player == 'human':
        return HumanPlayer()
    else:
        return MuZeroPlayer()


def create_game(nodes: int, complete_graph: bool) -> Game:
    graphs = generate_graphs(nodes)
    return GraphColoring(graphs, complete_graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuZero')

    mode_parsers = parser.add_subparsers(title='Modes')

    play_parser = mode_parsers.add_parser('play')
    play_parser.set_defaults(mode='play')
    player_choices = ['random', 'human', 'muzero']
    play_parser.add_argument('--p1', type=str, choices=player_choices, default='human',
                            help='Player 1')
    play_parser.add_argument('--p1-config', type=str, help='P1 config folder')
    play_parser.add_argument('--p2', type=str, choices=player_choices, default='human',
                            help='Player 2')
    play_parser.add_argument('--p2-config', type=str, help='P2 config folder')

    train_parser = mode_parsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    train_parser.add_argument('--exp-name', type=str, default='gc',
                                help='Experiment name')
    train_parser.add_argument('--gpu', action='store_true',
                                help='Whether to enable GPU (if available)')
    selfplay_args = train_parser.add_argument_group('Self-play arguments')
    selfplay_args.add_argument('--seed', type=int, default=0,
                                help='Seed')
    selfplay_args.add_argument('--workers', type=int, default=1,
                                help='Number of self-play workers')
    selfplay_args.add_argument('--max-moves', type=int, default=10,
                                help='Maximum number of moves to end the game early')
    selfplay_args.add_argument('--stacked-observations', type=int, default=1,
                                help='')
    selfplay_args.add_argument('--stack-action', action='store_true',
                                help='Whether to attach historical actions when stacking observations')
    selfplay_args.add_argument('--simulations', type=int, default=25,
                                help='Number of MCTS simulations')
    selfplay_args.add_argument('--gamma', type=float, default=1,
                                help='Discount factor')
    selfplay_args.add_argument('--root-dirichlet-alpha', type=float, default=0.1,
                                help='')
    selfplay_args.add_argument('--root-exploration-fraction', type=float, default=0.25,
                                help='')
    selfplay_args.add_argument('--c-base', type=float, default=19625,
                                help='')
    selfplay_args.add_argument('--c-init', type=float, default=1.25,
                                help='')

    network_args = train_parser.add_argument_group('Network training arguments')
    network_args.add_argument('--embedding-size', type=int, default=16,
                                help='')
    network_args.add_argument('--dynamics-layers', type=int, nargs='+', default=[8],
                                help='Hidden layers in dynamics network')
    network_args.add_argument('--reward-layers', type=int, nargs='+', default=[8],
                                help='Hidden layers in reward head')
    network_args.add_argument('--policy-layers', type=int, nargs='+', default=[8],
                                help='Hidden layers in policy head')
    network_args.add_argument('--value-layers', type=int, nargs='+', default=[8],
                                help='Hidden layers in value head')
    network_args.add_argument('--batch-size', type=int, default=32,
                                help='Mini-batch size')
    network_args.add_argument('--checkpoint-interval', type=int, default=10,
                                help='Checkpoint interval')
    network_args.add_argument('--buffer-size', type=int, default=3000,
                                help='Replay buffer size')
    network_args.add_argument('--td-steps', type=int, default=10,
                                help='Number of steps in the future to take into account for target value calculation')
    network_args.add_argument('--unroll-steps', type=int, default=5,
                                help='Number of unroll steps')
    network_args.add_argument('--training-steps', type=int, default=100000,
                                help='Number of training steps')
    network_args.add_argument('--lr', type=float, default=0.003,
                                help='Learning rate')
    network_args.add_argument('--weight-decay', type=float, default=1e-4,
                                help='Weight decay')
    network_args.add_argument('--support-limit', type=int, default=10,
                                help='Support limit')
    network_args.add_argument('--value-loss-weight', type=float, default=0.25,
                                help='Weight of value loss in total loss function')
    network_args.add_argument('--save-model', action='store_true',
                                help='Whether to save the model')

    for p in [play_parser, train_parser]:
        p.add_argument('--nodes', type=int, default=10, help='Number of nodes')
        p.add_argument('--complete-graph', action='store_true', help='Whether to use complete graph as input to the representation network')

    args = parser.parse_args()

    game = create_game(args.nodes, args.complete_graph)
    args.players = game.players
    args.observation_dim = game.observation_dim
    args.action_space = game.action_space
    args.visit_softmax_temperature_func = game.visit_softmax_temperature_func

    if args.mode == 'play':
        p1 = create_player(args.p1)
        p2 = create_player(args.p2)
        arena = Arena(p1, p2, game)
        arena.run(True)
    elif args.mode == 'train':
        args.device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'
        muzero = MuZero(game, args)
        muzero.train()
        