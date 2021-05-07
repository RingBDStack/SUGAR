# SUGAR

Code for "[SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism](https://arxiv.org/abs/2101.08170)"

## Overview

- train.py: the core of our model, including the structure and the process of training.
- env.py, QLearning.py: the code about RL method
- GCN.py, layers.py: including the basic layers we used in the main model.
- dataset/: including the dataset:MUTAG, DD, NCI1, NCI109, PTC_MR, ENZYMES, PROTEINS.
  - 'RAW/': the original data of the dataset
  - adj.npy: the biggest Adjacency Matrix built from dataset
  - graph_label.npy: the label of every sub_graph
  - sub_adj.npy: the Adjacency Matrix of subgraph through sampling
  - features.npy: the pre-processed features of each subgraph

## Datasets

- MUTAG: The MUTAG dataset consists of 188 chemical compounds divided into two
  classes according to their mutagenic effect on a bacterium.
- D&D: D&D is a dataset of 1178 protein structures (Dobson and Doig, 2003). Each protein is
  represented by a graph, in which the nodes are amino acids and two nodes are connected
  by an edge if they are less than 6 Angstroms apart. The prediction task is to classify
  the protein structures into enzymes and non-enzymes.
- NCI1&NCI109:NCI1 and NCI109 represent two balanced subsets of datasets of chemical compounds screened
  for activity against non-small cell lung cancer and ovarian cancer cell lines respectively
  (Wale and Karypis (2006) and http://pubchem.ncbi.nlm.nih.gov).
- ENZYMES: ENZYMES is a dataset of protein tertiary structures obtained from (Borgwardt et al., 2005)
  consisting of 600 enzymes from the BRENDA enzyme database (Schomburg et al., 2004).
  In this case the task is to correctly assign each enzyme to one of the 6 EC top-level
  classes.

## Setting

1. setting python env using pip install -r requirements.txt
2. cd ./dataset &python transform.py --dataset MUTAG
3.  python train.py(all the parameters could be viewed in the train.py)

## Parameters
````
     --dataset DATASET
     --num_info NUM_INFO
     --lr LR (learning_rate)
     --max_pool MAX_POOL
     --momentum MOMENTUM
     --num_epoch NUM_EPOCH
     --batch_size BATCH_SIZE
     --sg_encoder SG_ENCODER(GIN, GCN, GAT, SAGE)
     --MI_loss MI_LOSS
     --start_k START_K
 ````

## Reference
````
@inproceedings{sun2021sugar,
  title={SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism},
  author={Sun, Qingyun and Li, Jianxin and Peng, Hao and Wu, Jia and Ning, Yuanxing and Yu, Phillip S and He, Lifang},
  booktitle={Proceedings of the 2021 World Wide Web Conference},
  year={2021}
}
````
