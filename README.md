# Hierarchical Vector Quantized Graph Autoencoder with Annealing-Based Code Selection

Implementation for HQA-GAE


## Node Classification

```
python train.py --dataset <dataset_name> --task nc

```

`<dataset_name>` can be cora, citeseer, pubmed, physics, computers, cs, photo, ogbn-arxiv


## Link Prediction

```
python train.py --dataset <dataset_name> --task lp

```

`<dataset_name>` can be cora, citeseer, pubmed, physics, computers, cs, photo