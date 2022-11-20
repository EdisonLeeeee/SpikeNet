from distutils.version import LooseVersion

import gensim
import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec as _Word2Vec
from numba import njit
from sklearn import preprocessing


class DeepWalk:
    r"""Implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
    from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.
    """

    def __init__(self, dimensions: int = 64,
                 walk_length: int = 80,
                 walk_number: int = 10,
                 workers: int = 3,
                 window_size: int = 5,
                 epochs: int = 1,
                 learning_rate: float = 0.025,
                 negative: int = 1,
                 name: str = None,
                 seed: int = None):

        kwargs = locals()
        kwargs.pop("self")
        kwargs.pop("__class__", None)

        self.set_hyparas(kwargs)

    def set_hyparas(self, kwargs: dict):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.hyparas = kwargs

    def fit(self, graph: sp.csr_matrix):
        walks = RandomWalker(walk_length=self.walk_length,
                             walk_number=self.walk_number).walk(graph)
        sentences = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(sentences,
                         sg=1,
                         hs=0,
                         alpha=self.learning_rate,
                         iter=self.epochs,
                         size=self.dimensions,
                         window=self.window_size,
                         workers=self.workers,
                         negative=self.negative,
                         seed=self.seed)
        self._embedding = model.get_embedding()

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize:
            embedding = preprocessing.normalize(embedding)
        return embedding


class RandomWalker:
    """Fast first-order random walks in DeepWalk

    Parameters:
    -----------
    walk_number (int): Number of random walks. Default is 10.
    walk_length (int): Length of random walks. Default is 80.
    """

    def __init__(self, walk_length: int = 80, walk_number: int = 10):
        self.walk_length = walk_length
        self.walk_number = walk_number

    def walk(self, graph: sp.csr_matrix):
        walks = self.random_walk(graph.indices,
                                 graph.indptr,
                                 walk_length=self.walk_length,
                                 walk_number=self.walk_number)
        return walks

    @staticmethod
    @njit(nogil=True)
    def random_walk(indices,
                    indptr,
                    walk_length,
                    walk_number):
        N = len(indptr) - 1
        for _ in range(walk_number):
            for n in range(N):
                walk = [n]
                current_node = n
                for _ in range(walk_length - 1):
                    neighbors = indices[
                        indptr[current_node]:indptr[current_node + 1]]
                    if neighbors.size == 0:
                        break
                    current_node = np.random.choice(neighbors)
                    walk.append(current_node)

                yield walk


class Word2Vec(_Word2Vec):
    """A compatible version of Word2Vec"""

    def __init__(self, sentences=None, sg=0, hs=0, alpha=0.025, iter=5, size=100, window=5, workers=3, negative=5, seed=None, **kwargs):
        if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
            super().__init__(sentences,
                             size=size,
                             window=window,
                             min_count=0,
                             alpha=alpha,
                             sg=sg,
                             workers=workers,
                             iter=iter,
                             negative=negative,
                             hs=hs,
                             compute_loss=True,
                             seed=seed, **kwargs)

        else:
            super().__init__(sentences,
                             vector_size=size,
                             window=window,
                             min_count=0,
                             alpha=alpha,
                             sg=sg,
                             workers=workers,
                             epochs=iter,
                             negative=negative,
                             hs=hs,
                             compute_loss=True,
                             seed=seed, **kwargs)

    def get_embedding(self):
        if LooseVersion(gensim.__version__) <= LooseVersion("4.0.0"):
            embedding = self.wv.vectors[np.fromiter(
                map(int, self.wv.index2word), np.int32).argsort()]
        else:
            embedding = self.wv.vectors[np.fromiter(
                map(int, self.wv.index_to_key), np.int32).argsort()]

        return embedding
