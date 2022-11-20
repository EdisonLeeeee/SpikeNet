import math
import os.path as osp
from collections import defaultdict, namedtuple
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

Data = namedtuple('Data', ['x', 'edge_index'])


def standard_normalization(arr):
    n_steps, n_node, n_dim = arr.shape
    arr_norm = preprocessing.scale(np.reshape(arr, [n_steps, n_node * n_dim]), axis=1)
    arr_norm = np.reshape(arr_norm, [n_steps, n_node, n_dim])
    return arr_norm


def edges_to_adj(edges, num_nodes, undirected=True):
    row, col = edges
    data = np.ones(len(row))
    N = num_nodes
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    if undirected:
        adj = adj.maximum(adj.T)
    adj[adj > 1] = 1
    return adj


class Dataset:
    def __init__(self, name=None, root="./data"):
        self.name = name
        self.root = root
        self.x = None
        self.y = None
        self.num_features = None
        self.adj = []
        self.adj_evolve = []
        self.edges = []
        self.edges_evolve = []

    def _read_feature(self):
        filename = osp.join(self.root, self.name, f"{self.name}.npy")
        if osp.exists(filename):
            return np.load(filename)
        else:
            return None

    def split_nodes(
        self,
        train_size: float = 0.4,
        val_size: float = 0.0,
        test_size: float = 0.6,
        random_state: Optional[int] = None,
    ):
        val_size = 0. if val_size is None else val_size
        assert train_size + val_size + test_size <= 1.0

        y = self.y
        train_nodes, test_nodes = train_test_split(
            torch.arange(y.size(0)),
            train_size=train_size + val_size,
            test_size=test_size,
            random_state=random_state,
            stratify=y)

        if val_size:
            train_nodes, val_nodes = train_test_split(
                train_nodes,
                train_size=train_size / (train_size + val_size),
                random_state=random_state,
                stratify=y[train_nodes])
        else:
            val_nodes = None

        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes

    def split_edges(
        self,
        train_stamp: float = 0.7,
        train_size: float = None,
        val_size: float = 0.1,
        test_size: float = 0.2,
        random_state: int = None,
    ):

        if random_state is not None:
            torch.manual_seed(random_state)

        num_edges = self.edges[-1].size(-1)
        train_stamp = train_stamp if train_stamp >= 1 else math.ceil(len(self) * train_stamp)

        train_edges = torch.LongTensor(np.hstack(self.edges_evolve[:train_stamp]))
        if train_size is not None:
            assert 0 < train_size < 1
            num_train = math.floor(train_size * num_edges)
            perm = torch.randperm(train_edges.size(1))[:num_train]
            train_edges = train_edges[:, perm]

        num_val = math.floor(val_size * num_edges)
        num_test = math.floor(test_size * num_edges)
        testing_edges = torch.LongTensor(np.hstack(self.edges_evolve[train_stamp:]))
        perm = torch.randperm(testing_edges.size(1))

        assert num_val + num_test <= testing_edges.size(1)

        self.train_stamp = train_stamp
        self.train_edges = train_edges
        self.val_edges = testing_edges[:, perm[:num_val]]
        self.test_edges = testing_edges[:, perm[num_val:num_val + num_test]]

    def __getitem__(self, time_index: int):
        x = self.x[time_index]
        edge_index = self.edges[time_index]
        snapshot = Data(x=x, edge_index=edge_index)
        return snapshot

    def __next__(self):
        if self.t < len(self):
            snapshot = self.__getitem__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self

    def __len__(self):
        return len(self.adj)

    def __repr__(self):
        return self.name


class DBLP(Dataset):
    def __init__(self, root="./data", normalize=True):
        super().__init__(name='dblp', root=root)
        edges_evolve, self.num_nodes = self._read_graph()
        x = self._read_feature()
        y = self._read_label()

        if x is not None:
            if normalize:
                x = standard_normalization(x)
            self.num_features = x.shape[-1]
            self.x = torch.FloatTensor(x)

        self.num_classes = y.max() + 1

        edges = [edges_evolve[0]]
        for e_now in edges_evolve[1:]:
            e_last = edges[-1]
            edges.append(np.hstack([e_last, e_now]))

        self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
        self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
        self.edges = [torch.LongTensor(edge) for edge in edges]
        self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately

        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}.txt")
        d = defaultdict(list)
        N = 0
        with open(filename) as f:
            for line in f:
                x, y, t = line.strip().split()
                x, y = int(x), int(y)
                d[t].append((x, y))
                N = max(N, x)
                N = max(N, y)
        N += 1
        edges = []
        for time in sorted(d):
            row, col = zip(*d[time])
            edge_now = np.vstack([row, col])
            edges.append(edge_now)
        return edges, N

    def _read_label(self):
        filename = osp.join(self.root, self.name, "node2label.txt")
        nodes = []
        labels = []
        with open(filename) as f:
            for line in f:
                node, label = line.strip().split()
                nodes.append(int(node))
                labels.append(label)

        nodes = np.array(nodes)
        labels = LabelEncoder().fit_transform(labels)

        assert np.allclose(nodes, np.arange(nodes.size))
        return labels


def merge(edges, step=1):
    if step == 1:
        return edges
    i = 0
    length = len(edges)
    out = []
    while i < length:
        e = edges[i:i + step]
        if len(e):
            out.append(np.hstack(e))
        i += step
    print(f'Edges has been merged from {len(edges)} timestamps to {len(out)} timestamps')
    return out


class Tmall(Dataset):
    def __init__(self, root="./data", normalize=True):
        super().__init__(name='tmall', root=root)
        edges_evolve, self.num_nodes = self._read_graph()
        x = self._read_feature()

        y, labeled_nodes = self._read_label()
        # reindexing
        others = set(range(self.num_nodes)) - set(labeled_nodes.tolist())
        new_index = np.hstack([labeled_nodes, list(others)])
        whole_nodes = np.arange(self.num_nodes)
        mapping_dict = dict(zip(new_index, whole_nodes))
        mapping = np.vectorize(mapping_dict.get)(whole_nodes)
        edges_evolve = [mapping[e] for e in edges_evolve]

        edges_evolve = merge(edges_evolve, step=10)

        if x is not None:
            if normalize:
                x = standard_normalization(x)
            self.num_features = x.shape[-1]
            self.x = torch.FloatTensor(x)

        self.num_classes = y.max() + 1

        edges = [edges_evolve[0]]
        for e_now in edges_evolve[1:]:
            e_last = edges[-1]
            edges.append(np.hstack([e_last, e_now]))

        self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
        self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
        self.edges = [torch.LongTensor(edge) for edge in edges]
        self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately

        self.mapping = mapping
        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}.txt")
        d = defaultdict(list)
        N = 0
        with open(filename) as f:
            for line in tqdm(f, desc='loading edges'):
                x, y, t = line.strip().split()
                x, y = int(x), int(y)
                d[t].append((x, y))
                N = max(N, x)
                N = max(N, y)
        N += 1
        edges = []
        for time in sorted(d):
            row, col = zip(*d[time])
            edge_now = np.vstack([row, col])
            edges.append(edge_now)
        return edges, N

    def _read_label(self):
        filename = osp.join(self.root, self.name, "node2label.txt")
        nodes = []
        labels = []
        with open(filename) as f:
            for line in tqdm(f, desc='loading nodes'):
                node, label = line.strip().split()
                nodes.append(int(node))
                labels.append(label)

        labeled_nodes = np.array(nodes)
        labels = LabelEncoder().fit_transform(labels)
        return labels, labeled_nodes


class Patent(Dataset):
    def __init__(self, root="./data", normalize=True):
        super().__init__(name='patent', root=root)
        edges_evolve = self._read_graph()
        y = self._read_label()
        edges_evolve = merge(edges_evolve, step=2)
        x = self._read_feature()

        if x is not None:
            if normalize:
                x = standard_normalization(x)
            self.num_features = x.shape[-1]
            self.x = torch.FloatTensor(x)

        self.num_nodes = y.size
        self.num_features = x.shape[-1]
        self.num_classes = y.max() + 1

        edges = [edges_evolve[0]]
        for e_now in edges_evolve[1:]:
            e_last = edges[-1]
            edges.append(np.hstack([e_last, e_now]))

        self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
        self.adj_evolve = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve]
        self.edges = [torch.LongTensor(edge) for edge in edges]
        self.edges_evolve = edges_evolve  # list of np.ndarray, the edges in each timestamp exist separately

        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}_edges.json")
        time_edges = defaultdict(list)
        with open(filename) as f:
            for line in tqdm(f, desc='loading patent_edges'):
                # src nodeID, dst nodeID, date, src originalID, dst originalID
                src, dst, date, _, _ = eval(line)
                date = date // 1e4
                time_edges[date].append((src, dst))

        edges = []
        for time in sorted(time_edges):
            edges.append(np.transpose(time_edges[time]))
        return edges

    def _read_label(self):
        filename = osp.join(self.root, self.name, f"{self.name}_nodes.json")
        labels = []
        with open(filename) as f:
            for line in tqdm(f, desc='loading patent_nodes'):
                # nodeID, originalID, date, node class
                node, _, date, label = eval(line)
                date = date // 1e4
                labels.append(label - 1)
        labels = np.array(labels)
        return labels
