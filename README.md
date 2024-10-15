# Hierarchical Vector Quantized Graph Autoencoder with Annealing-Based Code Selection

Implementation for HQA-GAE

## Link Prediction

```
python train.py --dataset <dataset_name> --task lp

```

`<dataset_name>` can be cora, citeseer, pubmed, physics, computers, cs, photo

## Node Classification

```
python train.py --dataset <dataset_name> --task nc

```

`<dataset_name>` can be cora, citeseer, pubmed, physics, computers, cs, photo, ogbn-arxiv

For SVM on large datasets, such as ogbn-arxiv, you should use the `--using_cuml` option to speed up the test time.