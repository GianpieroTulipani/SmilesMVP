import os
from itertools import repeat

import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import subgraph, to_networkx


class Molecule3DMaskingDataset(InMemoryDataset):
    def __init__(self, root, dataset, mask_ratio, smiles_path,
                 transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.root = root
        self.dataset = dataset
        self.mask_ratio = mask_ratio

        # Load SMILES strings
        self.smiles_list = pd.read_csv(smiles_path, header=None, names=["SMILES"])["SMILES"].tolist()

        super(Molecule3DMaskingDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

        assert len(self.smiles_list) == (self.slices["x"].shape[0] - 1), \
            f"Mismatch: {len(self.smiles_list)} SMILES vs. {self.slices['x'].shape[0] - 1} graphs."

        print(f'Dataset: {self.dataset}, Total molecules: {len(self.smiles_list)}')

    def subgraph(self, data):
        """Applies subgraph masking to the molecular graph."""
        G = to_networkx(data)
        node_num, _ = data.x.size()
        sub_num = int(node_num * (1 - self.mask_ratio))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        # BFS Sampling
        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set(range(node_num)) - set(idx_sub))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(set(G.neighbors(idx_sub[-1]))) - set(idx_sub)

        idx_nondrop = sorted(idx_sub)

        edge_idx, edge_attr = subgraph(subset=idx_nondrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_attr,
                                       relabel_nodes=True,
                                       num_nodes=node_num)

        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.positions = data.positions[idx_nondrop]
        data.__num_nodes__, _ = data.x.shape
        return data

    def get(self, idx):
        """Retrieves the SMILES string and molecular graph at index `idx`."""
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.mask_ratio > 0:
            data = self.subgraph(data)

        smiles = self.smiles_list[idx]  # Get corresponding SMILES
        return smiles, data  # Return as a tuple

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def download(self):
        pass

    def process(self):
        pass


def collate_fn(batch):
    """
    Custom collate function to batch SMILES strings and molecular graphs.
    """
    smiles_batch, graphs = zip(*batch)
    batch_graph = Batch.from_data_list(graphs) 
    return smiles_batch, batch_graph
