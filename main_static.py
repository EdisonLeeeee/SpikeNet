import argparse
import os.path as osp
import time

import torch
import torch.nn as nn
from sklearn import metrics
from spikenet import dataset, neuron
from spikenet.layers import SAGEAggregator
from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)
from torch.utils.data import DataLoader
from torch_geometric.datasets import Flickr, Reddit
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm


class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, T=5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):

        super().__init__()

        tau = 1.0
        if sampler == 'rw':
            self.sampler = RandomWalkSampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
        elif sampler == 'sage':
            self.sampler = Sampler(add_selfloops(to_scipy_sparse_matrix(data.edge_index)))
        else:
            raise ValueError(sampler)

        del data.edge_index

        aggregators, snn = nn.ModuleList(), nn.ModuleList()

        for hid in hids:
            aggregators.append(SAGEAggregator(in_features, hid,
                                              concat=concat, bias=bias,
                                              aggr=aggr))

            if act == "IF":
                snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
            elif act == 'LIF':
                snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
            elif act == 'PLIF':
                snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
            else:
                raise ValueError(act)

            in_features = hid * 2 if concat else hid

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.snn = snn
        self.sizes = sizes
        self.T = T
        self.pooling = nn.Linear(T * in_features, out_features)

    def encode(self, nodes):
        spikes = []
        sizes = self.sizes
        x = data.x

        for time_step in range(self.T):
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            for size in sizes:
                nbr = self.sampler(nbr, size)
                num_nodes.append(nbr.size(0))
                h.append(x[nbr].to(device))

            for i, aggregator in enumerate(self.aggregators):
                self_x = h[:-1]
                neigh_x = []
                for j, n_x in enumerate(h[1:]):
                    neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))

                out = self.snn[i](aggregator(self_x, neigh_x))
                if i != len(sizes) - 1:
                    out = self.dropout(out)
                    h = torch.split(out, num_nodes[:-(i + 1)])

            spikes.append(out)
        spikes = torch.cat(spikes, dim=1)
        neuron.reset_net(self)
        return spikes

    def forward(self, nodes):
        spikes = self.encode(nodes)
        return self.pooling(spikes)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="flickr",
                    help="Datasets (Reddit and Flickr only). (default: Flickr)")
parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2],
                    help='Neighborhood sampling size for each layer. (default: [5, 2])')
parser.add_argument('--hids', type=int, nargs='+',
                    default=[512, 10], help='Hidden units for each layer. (default: [128, 10])')
parser.add_argument("--aggr", nargs="?", default="mean",
                    help="Aggregate function ('mean', 'sum'). (default: 'mean')")
parser.add_argument("--sampler", nargs="?", default="sage",
                    help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
parser.add_argument("--surrogate", nargs="?", default="sigmoid",
                    help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
parser.add_argument("--neuron", nargs="?", default="LIF",
                    help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
parser.add_argument('--batch_size', type=int, default=2048,
                    help='Batch size for training. (default: 1024)')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate for training. (default: 5e-3)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Smooth factor for surrogate learning. (default: 1.0)')
parser.add_argument('--T', type=int, default=15,
                    help='Number of time steps. (default: 15)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout probability. (default: 0.5)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true',
                    help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed for model. (default: 2022)')


try:
    args = parser.parse_args()
    args.split_seed = 42
    tab_printer(args)
except:
    parser.print_help()
    exit(0)

assert len(args.hids) == len(args.sizes), "must be equal!"

root = "data/"  # Specify your root path

if args.dataset.lower() == "reddit":
    dataset = Reddit(osp.join(root, 'Reddit'))
    data = dataset[0]
elif args.dataset.lower() == "flickr":
    dataset = Flickr(osp.join(root, 'Flickr'))
    data = dataset[0]
    
data.x = torch.nn.functional.normalize(data.x, dim=1)

set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

y = data.y.to(device)

train_loader = DataLoader(data.train_mask.nonzero().view(-1), pin_memory=False, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(data.val_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)
test_loader = DataLoader(data.test_mask.nonzero().view(-1), pin_memory=False, batch_size=10000, shuffle=False)


model = SpikeNet(dataset.num_features, dataset.num_classes, alpha=args.alpha,
                 dropout=args.dropout, sampler=args.sampler, T=args.T,
                 aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
                 hids=args.hids, act=args.neuron, bias=True).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()


def train():
    model.train()
    for nodes in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        loss_fn(model(nodes), y[nodes]).backward()
        optimizer.step()


@torch.no_grad()
def test(loader):
    model.eval()
    logits = []
    labels = []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    logits = logits.argmax(1)
    metric_macro = metrics.f1_score(labels, logits, average='macro')
    metric_micro = metrics.f1_score(labels, logits, average='micro')
    return metric_macro, metric_micro


best_val_metric = test_metric = 0
start = time.time()
for epoch in range(1, args.epochs + 1):
    train()
    val_metric, test_metric = test(val_loader), test(test_loader)
    if val_metric[1] > best_val_metric:
        best_val_metric = val_metric[1]
        best_test_metric = test_metric
    end = time.time()
    print(
        f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')

# save bianry node embeddings (spikes)
# emb = model.encode(torch.arange(data.num_nodes)).cpu()
# torch.save(emb, 'emb.pth')
