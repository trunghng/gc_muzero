# gc_muzero
An attempt to use MuZero, Gumbel MuZero with Graph Neural Nets on solving the [Graph coloring problem](https://en.wikipedia.org/wiki/Graph_coloring).

## Requirements
The project is running on Python 3.10. To install dependencies, run the following command
```bash
pip install -r requirements.txt
```

## Usage

### Graph dataset generation
Graph datasets are generated directly via `main.py` with `graph_generation` mode selected. The creation can be chosen with either necessary arguments or predefined configs.
```
usage: main.py graph_generation [-h] [--config-path CONFIG_PATH] [--nodes NODES] [--graphs GRAPHS]
                                [--graph-types {ER,BA,WS,LT} [{ER,BA,WS,LT} ...]]
                                [--chromatic-number CHROMATIC_NUMBER]
                                [--dataset-name DATASET_NAME]

options:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH
                        Path to the config file
  --nodes NODES         Number of nodes for each graph
  --graphs GRAPHS       Number of graphs for each type
  --graph-types {ER,BA,WS,LT} [{ER,BA,WS,LT} ...]
                        List of graph types
  --chromatic-number CHROMATIC_NUMBER
                        Chromatic number, for Leighton graph generation
  --dataset-name DATASET_NAME
                        Name of the dataset to save
```
For example, to generate a dataset from a config stored at `/configs/graph_generation/lt_n50k10.json` (i.e. Leighton graphs with chromatic number 10, each of which contains 50 nodes):
```bash
python main.py graph_generation --config-path configs/graph_generation/lt_n50k10.json
```

### Running experiments
Each experiment can also be run by calling `main.py`, choosing mode (`train` or `test`), and again, either with required arguments or with a predefined config file. For instance:
```bash
python main.py train --config-path configs/train/muzero.json
```

## Acknowledgements
The code is heavily inspired by these repos:
- [muzero-general](https://github.com/werner-duvaud/muzero-general)
- [mctx](https://github.com/google-deepmind/mctx)

## References
[1] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, David Silver et al. [Mastering Atari, Go, chess and shogi by planning with a learned model](https://doi.org/10.1038/s41586-020-03051-4). Nature 588, 604–609, 2020.  
[2] Ivo Danihelka, Arthur Guez, Julian Schrittwieser, David Silver. [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO). ICLR, 2022.  
[3] Leighton, Frank Thomson. [A Graph Coloring Algorithm for Large Scheduling Problems](https://doi.org/10.6028/jres.084.024). Journal of research of the National Bureau of Standards 84 6 (1979): 489-506.