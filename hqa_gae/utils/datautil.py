import math
import torch
import os
import torch_geometric.transforms as T
from torch.nn.functional import one_hot
from torch_geometric.utils import negative_sampling, to_undirected, add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, TUDataset
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
from torch_sparse import SparseTensor

class DataUtil(object):

    @staticmethod        
    def get_dataset(root: str, name: str, split="public", ratios=[0.6, 0.2, 0.2], **kwargs):
        assert split.lower() in ["public", "random"]
        name = name.lower()
        just_one_graph = True
        if name in {"cora", "citeseer", "pubmed"}:
            dataset = Planetoid(root, name, **kwargs)
        elif name in {'photo', 'computers'}:
            dataset = Amazon(root, name, **kwargs)
        elif name in {'cs', 'physics'}:
            dataset = Coauthor(root, name, **kwargs)
        elif name in {'ogbn-arxiv', 'ogbn-proteins', 'ogbn-mag'}:
            dataset = PygNodePropPredDataset(name=name, root=root, transform=T.ToUndirected(), **kwargs) 
            if name == "ogbn-proteins":
                adj_t = SparseTensor.from_edge_index(dataset.edge_index, dataset.edge_attr)
                dataset._data.x = adj_t.mean(dim=1)
            elif name == "ogbn-mag":
                data = Data(
                x=dataset[0].x_dict['paper'],
                edge_index=dataset[0].edge_index_dict[('paper', 'cites', 'paper')],
                y=dataset[0].y_dict['paper'])
                data = T.ToUndirected()(data)
                dataset._data = data
            dataset._data.y.squeeze_(1)
            if split == "public":
                split_idx = dataset.get_idx_split()
                if name == "ogbn-mag":
                    train_idx, valid_idx, test_idx = split_idx["train"]["paper"], split_idx["valid"]["paper"], split_idx["test"]["paper"]
                else:
                    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
                train_mask, val_mask, test_mask = [torch.zeros(dataset.x.size(0)).bool() for _ in range(3)]
                train_mask[train_idx] = True
                val_mask[valid_idx] = True
                test_mask[test_idx] = True
                dataset._data.train_mask = train_mask
                dataset._data.val_mask = val_mask
                dataset._data.test_mask = test_mask
        elif name in {"imdb-binary", "imdb-multi", "imdb-b", "imdb-m", "proteins", 
                      "collab", "mutag", "reddit-b", "reddit-binary", "nci1"}:
            if name == "imdb-b": name = "imdb-binary"
            elif name == "imdb-m": name = "imdb-multi"
            elif name == "reddit-b": name = "reddit-binary"
            just_one_graph = False
            name = name.upper()
            dataset = TUDataset(root, name, **kwargs)

            if dataset[0].x is None:

                x_path = os.path.join(root, name, "node_feats", "x.pt")
                slice_path = os.path.join(root, name, "node_feats", "slice.pt")
                if os.path.exists(x_path) and os.path.exists(slice_path):
                    feats = torch.load(x_path)
                    slice = torch.load(slice_path)
                else:
                    MAX_DEGREES = 400

                    slice = [0]
                    feature_dim = 0
                    for data in tqdm(dataset, desc="Calculating degrees"):
                        deg = degree(data.edge_index[0])
                        feature_dim = max(feature_dim, deg.max().item())

                    feature_dim = int(min(feature_dim, MAX_DEGREES)) + 1

                    feats = []

                    for data in tqdm(dataset, desc="Creating node features"):
                        deg = degree(data.edge_index[0], dtype=torch.long)
                        deg[deg > MAX_DEGREES] = MAX_DEGREES

                        feat = one_hot(deg, num_classes=feature_dim).float()
                        feats.append(feat)
                        slice.append(slice[-1] + data.num_nodes)
                    
                    feats = torch.cat(feats, dim=0)
                    slice = torch.tensor(slice)
                    os.makedirs(os.path.dirname(x_path), exist_ok=True)
                    torch.save(feats.data, x_path)
                    torch.save(slice.data, slice_path)

                dataset._data_list = None
                dataset._data.x = feats
                dataset.slices["x"] = slice

        else:
            raise ValueError(f"Unrecognized dataset: {name}")
        
        if split == "random" and just_one_graph:
            train_mask, val_mask, test_mask = DataUtil.mask_splits(dataset.x.shape[0], ratios)
            dataset._data.train_mask = train_mask
            dataset._data.val_mask = val_mask
            dataset._data.test_mask = test_mask

        return dataset
    

    @staticmethod
    def mask_splits(N, ratios=[0.6, 0.2, 0.2]):
        assert len(ratios) == 3, "ratios should be a list of 3 elements, stand for train, val, test ratio"
        ratios = torch.tensor(ratios)
        ptr = torch.round(torch.cumsum(ratios, 0) * N).type(torch.int64)
        assert ptr[2] <= N, "ratio sum should be less than 1"
        train_mask, val_mask, test_mask = [torch.zeros((N,), dtype=torch.bool) for _ in range(3)]
        shuffle_idxs = torch.randperm(N)
        train_mask[shuffle_idxs[:ptr[0]]] = True
        val_mask[shuffle_idxs[ptr[0]: ptr[1]]] = True
        test_mask[shuffle_idxs[ptr[1]:]] = True
        return train_mask, val_mask, test_mask

    @staticmethod
    def train_test_split_edges(edge_index, num_node=None, val_ratio: float = 0.05,
                               test_ratio: float = 0.1, is_undirected: bool = True):
        r"""Splits the edges of a :class:`torch_geometric.data.Data` object
        into positive and negative train/val/test edges.
        As such, it will replace the :obj:`edge_index` attribute with
        :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
        :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
        :obj:`test_pos_edge_index` attributes.

        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation edges.
                (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test edges.
                (default: :obj:`0.1`)

        :rtype: :class:`torch_geometric.data.Data`
        """

        # assert 'batch' not in data  # No batch-mode.

        num_nodes = edge_index.max().item() + 1 if num_node is None else num_node
        row, col = edge_index
        # Return upper triangular portion.
        if is_undirected:
            mask = row < col
            row, col = row[mask], col[mask]

        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        val_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        train_pos_edge_index = torch.stack([r, c], dim=0)

        if is_undirected:
            train_pos_edge_index = to_undirected(train_pos_edge_index)
            val_pos_edge_index = to_undirected(val_pos_edge_index)
            test_pos_edge_index = to_undirected(test_pos_edge_index)

        cum_edge_index, _ = add_self_loops(edge_index)
        val_neg_edge_index = negative_sampling(
            cum_edge_index, num_nodes=num_nodes,
            num_neg_samples=val_pos_edge_index.size(1),
            force_undirected=is_undirected)
        
        cum_edge_index = torch.cat([cum_edge_index, val_pos_edge_index], dim=1)
        test_neg_edge_index = negative_sampling(
            cum_edge_index, num_nodes=num_nodes,
            num_neg_samples=test_pos_edge_index.size(1),
            force_undirected=is_undirected)
        
        cum_edge_index = torch.cat([cum_edge_index, test_neg_edge_index], dim=1)
        train_neg_edge_index = negative_sampling(
            cum_edge_index, num_nodes=num_nodes,
            num_neg_samples=train_pos_edge_index.size(1),
            force_undirected=is_undirected)

        return {
            "pos": {
                "train":train_pos_edge_index,
                "val": val_pos_edge_index,
                "test": test_pos_edge_index,
            },
            "neg": {
                "train": train_neg_edge_index, 
                "val": val_neg_edge_index,
                "test": test_neg_edge_index
            }
        }