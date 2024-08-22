import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GraphConv, PNAConv
from game import GraphColoring, GameHistory
from utils import generate_erdos_renyi_graph
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import networkx as nx
import random


class GraphConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GraphConv(2, 16)
        self.conv2 = GraphConv(16, 16)
        self.conv3 = GraphConv(16, 16)

        self.mlp = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        print(x.shape, edge_index)
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))

        q = self.mlp(h)
        return h, q


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    graphs = [
        generate_erdos_renyi_graph(5),
        # generate_erdos_renyi_graph(10),
        # generate_erdos_renyi_graph(15)
    ]
    # game = GraphColoring(graphs)
    # game_histories = []

    # for _ in range(len(graphs)):
    #     game.reset()
    #     game.render()
    #     game_history = GameHistory(game)
    #     game_histories.append(game_history)


    game = GraphColoring(graphs, True)
    game.reset()

    data = Data(
        x=torch.tensor(game.observation(), dtype=torch.float32, requires_grad=False),
        edge_index=torch.tensor(list(game.graph.edges), dtype=torch.int64, requires_grad=False).transpose(0, 1),
        edge_attr=torch.tensor(game.edge_features, dtype=torch.int64, requires_grad=False)
    )

    model = GraphConvNet()
    h, q = model(data)
    print(h.shape, q)



    # datas = [
    #     Data(
    #         x=torch.tensor(game.observation(), dtype=torch.float, requires_grad=False),
    #         edge_index=torch.tensor(list(game.graph.edges), dtype=torch.long, requires_grad=False).transpose(0,1)
    #     )
    #     for game in games
    # ]

    # batch = Batch.from_data_list(datas)
    # print(type(batch), batch)
    # model = GraphConvNet()
    # h = model(batch)
    # print(h)

    # graph = generate_erdos_renyi_graph(7)
    # game = GraphColoring(graph)
    # game_history = GameHistory(game)
    # stacked_observations = [gh.stack_observations(-1, 1, game.action_space, False) for gh in game_histories]
    # batch = Batch.from_data_list(stacked_observations)
    # model = GraphConvNet()

    # print(batch)
    # h, q = model(batch)
    # print(h.shape, q.shape)
