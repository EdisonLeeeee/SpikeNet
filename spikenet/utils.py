import numpy as np
import scipy.sparse as sp
import torch
from sample_neighber import sample_neighber_cpu
from texttable import Texttable

try:
    import torch_cluster
except ImportError:
    torch_cluster = None


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def tab_printer(args):
    """Function to print the logs in a nice tabular format.
    
    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.
    
    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
    print(t.draw())

    
class Sampler:
    def __init__(self, adj_matrix: sp.csr_matrix):
        self.rowptr = torch.LongTensor(adj_matrix.indptr)
        self.col = torch.LongTensor(adj_matrix.indices)

    def __call__(self, nodes, size, replace=True):
        nbr = sample_neighber_cpu(self.rowptr, self.col, nodes, size, replace)
        return nbr
    
    
class RandomWalkSampler:
    def __init__(self, adj_matrix: sp.csr_matrix, p: float = 1.0, q: float = 1.0):
        self.rowptr = torch.LongTensor(adj_matrix.indptr)
        self.col = torch.LongTensor(adj_matrix.indices)
        self.p = p
        self.q = q
        assert torch_cluster, "Please install 'torch_cluster' first."

    def __call__(self, nodes, size, replace=True):
        nbr = torch.ops.torch_cluster.random_walk(self.rowptr, self.col, nodes, size, self.p, self.q)[0][:, 1:] 
        return nbr


def eliminate_selfloops(adj_matrix):
    """eliminate selfloops for adjacency matrix.

    >>>eliminate_selfloops(adj) # return an adjacency matrix without selfloops

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array

    Returns
    -------
    Single Scipy sparse matrix or Numpy matrix.

    """
    if sp.issparse(adj_matrix):
        adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format='csr')
        adj_matrix.eliminate_zeros()
    else:
        adj_matrix = adj_matrix - np.diag(adj_matrix)
    return adj_matrix


def add_selfloops(adj_matrix: sp.csr_matrix):
    """add selfloops for adjacency matrix.

    >>>add_selfloops(adj) # return an adjacency matrix with selfloops

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array

    Returns
    -------
    Single sparse matrix or Numpy matrix.

    """
    adj_matrix = eliminate_selfloops(adj_matrix)

    return adj_matrix + sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='csr')
