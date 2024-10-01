import argparse
from argparse import RawTextHelpFormatter
import json
import os.path as osp
import sys
from types import SimpleNamespace

import torch

from games.game import Game
from games.graph_coloring import GraphColoring
from muzero import MuZero
from utils.graph_utils import generate_graphs, save_dataset, load_dataset


def create_game(args) -> Game:
    graphs = load_dataset(args.dataset_path)
    if args.mode == 'train':
        return GraphColoring(graphs, args.complete_graph)
    elif args.mode == 'test':
        return GraphColoring(graphs)


def validate_args(parser, args) -> None:
    def validate1(required_tags):
        unspecified_tags = [k for k in required_tags if required_tags[k] is None]
        if unspecified_tags:
            parser.error(f'{", ".join(unspecified_tags)} must be specified.')

    def validate2(tag_value, value_wanted, tag, tag_dict):
        unspecified_tags = [k for k in tag_dict if tag_dict[k] is None]
        if tag_value == value_wanted and unspecified_tags:
            parser.error(f'{", ".join(unspecified_tags)} must be specified \
                when {tag}={value_wanted}.')

    if args.mode == 'graph_generation':
        validate1({
            '--nodes': args.nodes,
            '--graphs': args.graphs,
            '--graph-types': args.graph_types,
            '--dataset-name': args.dataset_name
        })
        if 'LT' in args.graph_types and args.chromatic_number is None:
            parser.error('--chromatic-number must be specified for Leighton graphs.')
        if args.chromatic_number is not None and args.nodes % args.chromatic_number != 0:
            parser.error('--chromatic-number must be a factor of --nodes.')
    else:
        validate1({
            '--workers': args.workers,
            '--seed': args.seed,
            '--max-moves': args.max_moves,
            '--simulations': args.simulations,
            '--gamma': args.gamma,
            '--dataset-path': args.dataset_path
        })
        if args.mode == 'train':
            validate1({
                '--n-stacked-observations': args.n_stacked_observations,
                '--batch-size': args.batch_size,
                '--checkpoint-interval': args.checkpoint_interval,
                '--buffer-size': args.buffer_size,
                '--td-steps': args.td_steps,
                '--unroll-steps': args.unroll_steps,
                '--training-steps': args.training_steps,
                '--lr': args.lr,
                '--weight-decay': args.weight_decay,
                '--lr-decay-rate': args.lr_decay_rate,
                '--lr-decay-steps': args.lr_decay_steps,
                '--support-limit': args.support_limit,
                '--value-loss-weight': args.value_loss_weight,
                '--reanalyse-workers': args.reanalyse_workers,
                '--target-network-update-freq': args.target_network_update_freq
            })
        else:
            validate1({'--tests': args.tests})

        validate2(args.gumbel, True, '--gumbel', {
            '--max-considered-actions': args.max_considered_actions,
            '--c-visit': args.c_visit,
            '--c-scale': args.c_scale,
            '--gumbel-scale': args.gumbel_scale
        })
        validate2(args.gumbel, False, '--gumbel', {
            '--dirichlet-alpha': args.dirichlet_alpha,
            '--exploration-frac': args.exploration_frac,
            '--c-base': args.c_base,
            '--c-init': args.c_init
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Coloring with MuZero')

    mode_parsers = parser.add_subparsers(title='Modes')
    train_parser = mode_parsers.add_parser(
        'train', formatter_class=RawTextHelpFormatter
    )
    test_parser = mode_parsers.add_parser(
        'test', formatter_class=RawTextHelpFormatter
    )
    graph_gen_parser = mode_parsers.add_parser('graph_generation')

    for p in [train_parser, test_parser, graph_gen_parser]:
        p.add_argument('--config-path', type=str,
                       help='Path to the config file')

    for p in [train_parser, test_parser]:
        p.add_argument('--exp-name', type=str, default='gc',
                       help='Experiment name')
        p.add_argument('--workers', type=int, default=2,
                       help='Number of self-play workers')
        p.add_argument('--seed', type=int, default=0,
                       help='Seed for RNG')
        p.add_argument('--max-moves', type=int, default=50,
                       help='Maximum number of moves to end the game early')
        p.add_argument('--simulations', type=int, default=75,
                       help='Number of MCTS simulations')
        p.add_argument('--gamma', type=float, default=1,
                       help='Discount factor')
        p.add_argument('--gumbel', action='store_true',
                       help='Whether to use Gumbel MuZero')
        p.add_argument('--max-considered-actions', type=int, default=16,
                       help='Maximum number of actions sampled without \
                       replacement in Gumbel MuZero')
        p.add_argument('--c-visit', type=int, default=50,
                       help='')
        p.add_argument('--c-scale', type=float, default=1.0,
                       help='')
        p.add_argument('--gumbel-scale', type=float, default=1.0,
                       help='')
        p.add_argument('--dirichlet-alpha', type=float, default=0.25,
                       help='')
        p.add_argument('--exploration-frac', type=float, default=0.25,
                       help='')
        p.add_argument('--c-base', type=float, default=19625,
                       help='')
        p.add_argument('--c-init', type=float, default=1.25,
                       help='')
        p.add_argument('--logdir', type=str,
                       help='Path to the log directory, which stores \
                       model file, config file, etc')
        p.add_argument('--dataset-path', type=str,
                       help='Path to the stored graph dataset')

    train_parser.add_argument('--gpu', action='store_true',
                              help='Whether to enable GPU (if available)')
    train_parser.add_argument('--complete-graph', action='store_true',
                              help='Whether to use complete graph as input \
                              to the representation network')
    train_parser.add_argument('--n-stacked-observations', type=int, default=1,
                              help='')
    train_parser.add_argument('--stack-action', action='store_true',
                              help='Whether to attach historical actions \
                              when stacking observations')
    train_parser.add_argument('--embedding-size', type=int, default=32,
                              help='')
    train_parser.add_argument('--dynamics-layers', type=int, nargs='+', default=[16],
                              help='Hidden layers in dynamics network')
    train_parser.add_argument('--reward-layers', type=int, nargs='+', default=[16],
                              help='Hidden layers in reward head')
    train_parser.add_argument('--policy-layers', type=int, nargs='+', default=[16],
                              help='Hidden layers in policy head')
    train_parser.add_argument('--value-layers', type=int, nargs='+', default=[16],
                              help='Hidden layers in value head')
    train_parser.add_argument('--batch-size', type=int, default=128,
                              help='Mini-batch size')
    train_parser.add_argument('--checkpoint-interval', type=int, default=10,
                              help='Checkpoint interval')
    train_parser.add_argument('--buffer-size', type=int, default=3000,
                              help='Replay buffer size')
    train_parser.add_argument('--td-steps', type=int, default=50,
                              help='Number of steps in the future to take into \
                              account for target value calculation')
    train_parser.add_argument('--unroll-steps', type=int, default=5,
                              help='Number of unroll steps')
    train_parser.add_argument('--training-steps', type=int, default=100000,
                              help='Number of training steps')
    train_parser.add_argument('--lr', type=float, default=0.003,
                              help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4,
                              help='Weight decay')
    train_parser.add_argument('--lr-decay-rate', type=float, default=0.9,
                              help='Decay rate, used for exponential \
                              learning rate schedule')
    train_parser.add_argument('--lr-decay-steps', type=int, default=10000,
                              help='Number of decay steps, used for exponential \
                              learning rate schedule')
    train_parser.add_argument('--support-limit', type=int, default=4,
                              help='Support limit')
    train_parser.add_argument('--value-loss-weight', type=float, default=0.25,
                              help='Weight of value loss in total loss function')
    train_parser.add_argument('--reanalyse-workers', type=int, default=1,
                              help='Number of reanalyse workers')
    train_parser.add_argument('--target-network-update-freq', type=int, default=5,
                              help='Target network update frequency, used in \
                              Reanalyse to provide a fresher, stable target for \
                              the value function')
    train_parser.add_argument('--mcts-target-value', action='store_true',
                              help='Whether to use value function obtained from \
                              re-executing MCTS in Reanalyse as target for training')
    train_parser.set_defaults(mode='train')

    test_parser.add_argument('--tests', type=int, default=100,
                             help='Number of games for testing')
    test_parser.add_argument('--render', action='store_true',
                             help='Whether to render each game during testing')
    test_parser.add_argument('--player', type=str, choices=['random', 'human', 'greedy'],
                             help='Other player to test')
    test_parser.set_defaults(mode='test')

    graph_gen_parser.add_argument('--nodes', type=int,
                                  help='Number of nodes for each graph')
    graph_gen_parser.add_argument('--graphs', type=int,
                                  help='Number of graphs for each type')
    graph_gen_parser.add_argument('--graph-types', type=str, nargs='+', 
                                  choices=['ER', 'BA', 'WS', 'LT'],
                                  help='List of graph types')
    graph_gen_parser.add_argument('--chromatic-number', type=int,
                                  help='Chromatic number, for Leighton graph generation')
    graph_gen_parser.add_argument('--dataset-name', type=str,
                                  help='Name of the dataset to save')
    graph_gen_parser.set_defaults(mode='graph_generation')

    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        if hasattr(args, 'comment'):
            del args.comment
    else:
        validate_args(parser, args)

    if args.mode == 'train':
        if hasattr(args, 'logdir') and args.logdir is not None:
            try:
                with open(osp.join(args.logdir, 'config.json')) as f:
                    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
                config.logdir = args.logdir
                args = config
            except FileNotFoundError:
                print('Log directory not found')

        game = create_game(args)
        args.observation_dim = game.observation_dim
        args.n_actions = game.n_actions
        args.visit_softmax_temperature_func = game.visit_softmax_temperature_func
        args.device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'
        agent = MuZero(game, args)
        agent.train()
    elif args.mode == 'test':
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
        args.n_actions = game.n_actions
        args.n_stacked_observations = config['n_stacked_observations']
        args.embedding_size = config['embedding_size']
        args.dynamics_layers = config['dynamics_layers']
        args.reward_layers = config['reward_layers']
        args.policy_layers = config['policy_layers']
        args.value_layers = config['value_layers']
        args.support_limit = config['support_limit']
        args.stack_action = config['stack_action']
        agent = MuZero(game, args)
        agent.test()
    else:
        graphs = generate_graphs(
            args.nodes, args.graphs, args.graph_types, args.chromatic_number
        )
        save_dataset(args.dataset_name, graphs)
        