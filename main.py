import argparse
from argparse import RawTextHelpFormatter
import json
import os.path as osp
import sys

import torch

from game import Game, GraphColoring
from graph_utils import generate_graphs, save_dataset, load_dataset
from muzero import MuZero
from utils import generate_graphs, save_dataset, load_dataset


def create_game(args) -> Game:
    if args.load_graphs:
        graphs = load_dataset(args.graphs_path)
    else:
        graphs = generate_graphs(args.nodes, args.graph_types, args.graphs, args.chromatic_number)
        if args.save_graphs:
            save_dataset(args.savedir, graphs)
    if args.mode == 'train':
        return GraphColoring(graphs, args.complete_graph)
    else:
        return GraphColoring(graphs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Coloring with MuZero')

    mode_parsers = parser.add_subparsers(title='Modes')
    train_parser = mode_parsers.add_parser('train', formatter_class=RawTextHelpFormatter)
    train_parser.set_defaults(mode='train')
    test_parser = mode_parsers.add_parser('test', formatter_class=RawTextHelpFormatter)
    test_parser.set_defaults(mode='test')

    for p in [train_parser, test_parser]:
        p.add_argument('--nodes', type=int, default=10,
                       help='List of node amounts, for graph dataset generation')
        p.add_argument('--graph-types', type=str, nargs='+', choices=['ER', 'BA', 'WS', 'LT'], default=['ER'],
                       help='Type of graphs in the dataset')
        p.add_argument('--graphs', type=int, default=100,
                       help='Number of graphs for each type in the dataset')
        p.add_argument('--chromatic-number', type=int,
                       help='Chromatic number, for Leighton graph generation')
        p.add_argument('--save-graphs', action='store_true',
                       help='Whether to save the generated graph dataset')
        p.add_argument('--savedir', type=str,
                       help='Directory to save the graph dataset')
        p.add_argument('--load-graphs', action='store_true',
                       help='Whether to load the graph dataset')
        p.add_argument('--graphs-path', type=str,
                       help='Path to the graph dataset')
        p.add_argument('--exp-name', type=str, default='gc',
                       help='Experiment name')
        p.add_argument('--seed', type=int, default=0,
                       help='Seed for RNG')
        p.add_argument('--max-moves', type=int, default=9,
                       help='Maximum number of moves to end the game early')
        p.add_argument('--simulations', type=int, default=25,
                       help='Number of MCTS simulations')
        p.add_argument('--gamma', type=float, default=1,
                       help='Discount factor')
        p.add_argument('--root-dirichlet-alpha', type=float, default=0.2,
                       help='')
        p.add_argument('--root-exploration-fraction', type=float, default=0.25,
                       help='')
        p.add_argument('--c-base', type=float, default=19625,
                       help='')
        p.add_argument('--c-init', type=float, default=1.25,
                       help='')
        p.add_argument('--logdir', type=str,
                       help='Path to the log directory, which stores model file, config file, etc')

    train_parser.add_argument('--workers', type=int, default=2,
                              help='Number of self-play workers')
    train_parser.add_argument('--gpu', action='store_true',
                              help='Whether to enable GPU (if available)')
    train_parser.add_argument('--complete-graph', action='store_true',
                              help='Whether to use complete graph as input to the representation network')
    train_parser.add_argument('--stacked-observations', type=int, default=1,
                              help='')
    train_parser.add_argument('--stack-action', action='store_true',
                              help='Whether to attach historical actions when stacking observations')
    train_parser.add_argument('--embedding-size', type=int, default=16,
                              help='')
    train_parser.add_argument('--dynamics-layers', type=int, nargs='+', default=[8],
                              help='Hidden layers in dynamics network')
    train_parser.add_argument('--reward-layers', type=int, nargs='+', default=[8],
                              help='Hidden layers in reward head')
    train_parser.add_argument('--policy-layers', type=int, nargs='+', default=[8],
                              help='Hidden layers in policy head')
    train_parser.add_argument('--value-layers', type=int, nargs='+', default=[8],
                              help='Hidden layers in value head')
    train_parser.add_argument('--batch-size', type=int, default=32,
                              help='Mini-batch size')
    train_parser.add_argument('--checkpoint-interval', type=int, default=10,
                              help='Checkpoint interval')
    train_parser.add_argument('--buffer-size', type=int, default=3000,
                              help='Replay buffer size')
    train_parser.add_argument('--td-steps', type=int, default=10,
                              help='Number of steps in the future to take into account for target value calculation')
    train_parser.add_argument('--unroll-steps', type=int, default=5,
                              help='Number of unroll steps')
    train_parser.add_argument('--training-steps', type=int, default=100000,
                              help='Number of training steps')
    train_parser.add_argument('--lr', type=float, default=0.003,
                              help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4,
                              help='Weight decay')
    train_parser.add_argument('--lr-decay-rate', type=float, default=0.9,
                              help='Decay rate, used for exponential learning rate schedule')
    train_parser.add_argument('--lr-decay-steps', type=int, default=10000,
                              help='Number of decay steps, used for exponential learning rate schedule')
    train_parser.add_argument('--support-limit', type=int, default=10,
                              help='Support limit')
    train_parser.add_argument('--value-loss-weight', type=float, default=0.25,
                              help='Weight of value loss in total loss function')
    train_parser.add_argument('--reanalyse-workers', type=int, default=1,
                              help='Number of reanalyse workers')
    train_parser.add_argument('--target-network-update-freq', type=int, default=1,
                              help='Target network update frequency, used in Reanalyse to provide a '
                              'fresher, stable target for the value function')
    train_parser.add_argument('--mcts-target-value', action='store_true',
                              help='Whether to use value function obtained from re-executing MCTS in '
                              'Reanalyse as target for training')

    test_parser.add_argument('--tests', type=int, default=100,
                             help='Number of games for testing')
    test_parser.add_argument('--render', action='store_true',
                             help='Whether to render each game during testing')
    args = parser.parse_args()

    if args.chromatic_number is not None and args.nodes % args.chromatic_number != 0:
        parser.error('Argument --chromatic-number when being specified must be a divisor of --nodes.')
    if (args.save_graphs and args.savedir is None) or (not args.save_graphs and args.savedir is not None):
        parser.error('Arguments --save-graphs and --savedir must be specified together.')
    if (args.load_graphs and args.graphs_path is None) or (not args.load_graphs and args.graphs_path is not None):
        parser.error('Arguments --graphs-path and --graphs-path must be specified together.')

    if args.mode == 'train':
        if args.logdir is not None:
            try:
                with open(osp.join(args.logdir, 'config.json')) as f:
                    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
                config.logdir = args.logdir
                args = config
            except FileNotFoundError:
                print('Log directory not found')

        game = create_game(args)
        args.observation_dim = game.observation_dim
        args.action_space_size = game.action_space_size
        args.visit_softmax_temperature_func = game.visit_softmax_temperature_func
        args.device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'
        agent = MuZero(game, args)
        agent.train()
    else:
        try:
            with open(osp.join(args.logdir, 'config.json')) as f:
                config = json.load(f)
        except TypeError:
            print('--logdir tag must be defined')
            sys.exit(0)
        except FileNotFoundError:
            print('Log directory not found')
            sys.exit(0)

        game = create_game(args)
        args.observation_dim = game.observation_dim
        args.action_space_size = game.action_space_size
        args.stacked_observations = config['stacked_observations']
        args.embedding_size = config['embedding_size']
        args.dynamics_layers = config['dynamics_layers']
        args.reward_layers = config['reward_layers']
        args.policy_layers = config['policy_layers']
        args.value_layers = config['value_layers']
        args.support_limit = config['support_limit']
        args.stack_action = config['stack_action']
        agent = MuZero(game, args)
        agent.test()
        