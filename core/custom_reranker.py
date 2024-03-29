"""
A custom reranker model (custom = not based on any published paper)

Attributes:
    BASE_DIR (str): Absolute path to application's base directory
    MODELS_DIR (str): Absolute path to /assets/ directory
    embeddings (GloveWordEmbeddings): Word vectors
    sifs (dict): Smooth inverse frequencies of words
"""

import math
import json
from pathlib import Path
import numba
import numpy as np
from nltk.tokenize import RegexpTokenizer

from core.reranker import Ranker

BASE_DIR = str(Path(__file__).parent.parent.resolve())
ASSETS_DIR = f"{BASE_DIR}/assets/"

# pylint: disable=too-many-instance-attributes,too-many-arguments

class GloveWordEmbeddings:
    """Glove word embeddings"""

    def __init__(self):
        """Initialize"""
        self._assets_dir = ASSETS_DIR
        self._vocab_file = f"{self._assets_dir}/glove-vocab.json"
        self._dict_file = f"{self._assets_dir}/glove-dictionary.json"
        self._dfs_file = f"{self._assets_dir}/dfs.json"
        self._embs_file = f"{self._assets_dir}/glove-We.npy"
        self._vocab = None
        self._dictionary = None
        self._dfs = None
        self.sifs = None
        self._embs = None
        self._dims = None
        self._load()

    def _load(self):
        """Load the embeddings data from the disk
        """
        with open(self._vocab_file, "r", encoding="utf-8") as file:
            self._vocab = json.load(file)
        with open(self._dict_file, "r", encoding="utf-8") as file:
            self._dictionary = json.load(file)
        with open(self._dfs_file, "r", encoding="utf-8") as file:
            self._dfs = json.load(file)
        self._embs = np.load(self._embs_file)
        self.sifs = {word: self.df2sif(word, self._dfs) for word in self._dfs}
        self._dims = self._embs.shape[1]

    @staticmethod
    def df2sif(word, dfs):
        """Return SIF (smooth inverse frequency) of a given word

        Args:
            word (str): Word
            dfs (dict): Document frequencies of words {"word": df}

        Returns:
            float: SIF value
        """
        n = dfs[word]
        N = dfs["the"]
        p = n / N
        a = 0.01
        w = a / (a + p)
        return w

    def __len__(self):
        """Return the number of words for which vectors are available

        Returns:
            int: Word count
        """
        return self._embs.shape[0]

    def __getitem__(self, item):
        """Get a word vectors

        Args:
            item (str): Word

        Returns:
            List[float]: Word vector
        """
        assert isinstance(item, (int, str))
        if isinstance(item, int):
            return self._embs[item]
        if isinstance(item, str):
            if not item in self._dictionary:
                item = "<unk>"
            return self._embs[self._dictionary[item]]
        raise TypeError("Invalid item type")

    def get_sif(self, word):
        """Return SIF for a given word

        Args:
            word (str): Word

        Returns:
            float: SIF value
        """
        return self.sifs.get(word, 1.0)


embeddings = GloveWordEmbeddings()
sifs = embeddings.sifs


class VectorSequence:

    """A wrapper around a list of vectors"""

    def __init__(self, labels, vectors):
        """Initalize

        Args:
            labels (List[str]): An identifier for each vector
            vectors (List[List[float]]): A 2d matrix of floats (list of vectors)
        """
        self._labels = labels
        self._sequence = np.array(vectors)
        self._n = len(vectors)
        self._dims = self._sequence.shape[1]
        self._fixed_length = None
        self._default_interaction = Interaction()
        self._default_interaction.metric = "cosine"
        self._default_interaction.amplify = False
        self._default_interaction.reinforce = True
        self._default_interaction.context = True
        self._default_interaction.window = 5

    @property
    def labels(self):
        """Return the list of labels

        Returns:
            list: Labels
        """
        return self._labels

    def __repr__(self):
        """String representation

        Returns:
            str: String representation
        """
        text = f"VectorSequence: {len(self._labels)} labels, {len(self._sequence)} vectors;"
        text += f' Labels: {", ".join(self._labels[:5])}'
        text += ", ..." if self._labels[5:] else ""
        return text

    def weigh(self, weights):
        """Multiply vectors with weights

        Args:
            weights (dict or list): Weights corresponding to vectors or labels
        """

        if isinstance(weights, dict):
            W = np.array([weights[label] for label in self._labels])
        if isinstance(weights, list):
            W = np.array(weights)
        if isinstance(weights, np.ndarray):
            W = weights

        assert len(W) == len(self._sequence)
        W = W.reshape(1, -1)
        self._sequence *= W.T

    @property
    def redundancy_vector(self):
        """Baseline for interaction with self"""
        interactions = self._default_interaction.interact(self, self)
        interactions = np.tril(interactions.matrix, -1)
        return np.max(interactions, axis=1)

    @property
    def matrix(self):
        """Vector matrix

        Returns:
            np.ndarray: The matrix
        """
        if self._fixed_length is None:
            return self._sequence
        if self._n > self._fixed_length:
            return self._truncated
        return self._padded

    @property
    def _truncated(self):
        """Make fixed length by trunction"""
        return self._sequence[: self._fixed_length]

    @property
    def _padded(self):
        """Make fixed length by padding"""
        r = self._fixed_length - self._n
        shape = (r, self._dims)
        padding = np.zeros(shape)
        return np.concatenate((self._sequence, padding))

    def set_length(self, n):
        """Set a fixed length"""
        self._fixed_length = n

    @property
    def normalized_matrix(self):
        """Return a matrix with noramlized (unit) vectors"""
        row_magnitudes = np.sqrt(
            np.sum(self._sequence * self._sequence, axis=1, keepdims=True)
        )
        row_magnitudes += np.finfo(float).eps
        return self._sequence / row_magnitudes


class Interaction:

    """Creates an interaction matrix between two vector sequences"""

    def __init__(self, metric="cosine", context=False, amplify=False, reinforce=False, window=5):
        """Initialize

        Args:
            metric (str, optional): `cosine` or `dot` or `euclidean`
            context (bool, optional): Whether to apply context operation
            amplify (bool, optional): Whether to apply amplification
            reinforce (bool, optional): Whether to apply reinforcement
            window (int, optional): Window size for interaction
        """
        self.metric = metric
        self.context = context
        self.amplify = amplify
        self.reinforce = reinforce
        self.window_size = window

    def _context_sequence(self, vector_seq):
        """Return context sequence"""
        M = vector_seq.matrix
        C = np.zeros(M.shape)
        C *= np.array(
            [sifs[word] if word in sifs else 1.0 for word in vector_seq.labels]
        ).reshape((-1, 1))
        r = min(len(M - 1), self.window_size + 1)
        for i in range(1, r):
            C[i:, :] += M[:-i, :]
            C[:-i, :] += M[i:, :]
        return C

    def interact(self, vector_seq_A, vector_seq_B):
        """Create interaction matrix"""
        A = vector_seq_A.matrix
        B = vector_seq_B.matrix
        I = self.interaction_fn(A, B)
        I = self._amplifier(I) if self.amplify else I

        if not self.context:
            return InteractionMatrix(I)

        Ac = self._context_sequence(vector_seq_A)
        Bc = self._context_sequence(vector_seq_B)
        Ic = self.interaction_fn(Ac, Bc)
        Ic = self._amplifier(Ic) if self.amplify else Ic

        if not self.reinforce:
            return InteractionMatrix(I + Ic)

        M = self._reinforce(I, Ic)
        return InteractionMatrix(M)

    @property
    def interaction_fn(self):
        """Function used for creating interaction between two vectors"""
        if self.metric == "cosine":
            return self._cosine_interaction
        if self.metric == "dot":
            return self._dot_interaction
        if self.metric == "euclidean":
            return self._euclidean_interaction
        raise Exception("Invalid metric value")

    @staticmethod
    def _normalize_rows(M):
        """Normalize vectors to convert them to unit vectors"""
        row_magnitudes = np.sqrt(np.sum(M * M, axis=1, keepdims=True))
        row_magnitudes += np.finfo(float).eps
        return M / row_magnitudes

    @staticmethod
    def _reinforce(A, B):
        """Apply reinforcement"""
        return 0.25 * (A + B + 2 * (A * B))

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)])
    def _amplifier(x):
        """Apply amplification"""
        return 1 / (1 + (3.2 * math.exp(-7.5 * (x - 0.46))))
    
    @staticmethod
    def _dot_interaction(A, B):
        """Compute dot interaction"""
        return np.matmul(A, B.T)

    def _cosine_interaction(self, A, B):
        """Compute cosine interaction"""
        An = self._normalize_rows(A)
        Bn = self._normalize_rows(B)
        return self._dot_interaction(An, Bn)

    @staticmethod
    def _euclidean_interaction(A, B):
        """Compute euclidean interaction"""
        s = 0.0
        diffs = [a-b for a in A for b in B]
        for diff in diffs:
            s += diff * diff
        return np.sqrt(s)


class InteractionMatrix:

    """A matrix of interaction between two vector sequences
    """

    def __init__(self, I):
        """Initialize

        Args:
            I (List[List[floats]]): Interaction matrix
        """
        self._matrix = I

    def maxpool(self, direction="horizontal"):
        """Max pooling

        Args:
            direction (str, optional): "horizontal" or "vertical"

        Returns:
            list: Max pooling output
        """
        axis = 1 if direction == "horizontal" else 0
        return np.max(self._matrix, axis=axis)

    @property
    def matrix(self):
        """Return interaction matrix as numpy array"""
        return np.copy(self._matrix)


class CustomRanker(Ranker):

    """Custom model for scoring similarity between a query & a document
    """

    def __init__(self):
        """Initialize
        """
        self._interaction = Interaction()
        self._interaction.metric = "cosine"
        self._interaction.amplify = True
        self._interaction.reinforce = False
        self._interaction.context = False
        self._interaction.window = 10
        super().__init__("similarity")

    def score(self, query, document):
        """Get a numerical similarity score between the query and document

        Args:
            query (str): Query
            doc (str): Document

        Returns:
            float: Similarity score between query and document
        """
        qry_tokens = RegexpTokenizer(r"\w+").tokenize(query.lower())
        doc_tokens = RegexpTokenizer(r"\w+").tokenize(document.lower())

        Q = VectorSequence(qry_tokens, [embeddings[token] for token in qry_tokens])
        D = VectorSequence(doc_tokens, [embeddings[token] for token in doc_tokens])
        query_term_matches = self._interaction.interact(Q, D).maxpool()

        query_term_weights = [
            (sifs[word] if word in sifs else 1.0) for word in qry_tokens
        ]
        query_term_weights *= 1 - Q.redundancy_vector
        query_term_matches *= query_term_weights
        score = query_term_matches.sum()

        if query != document:
            score /= self.score(query, query)
            score /= self._calc_penalty_factor(qry_tokens, doc_tokens)
        return score

    @staticmethod
    def _calc_penalty_factor(qry_tokens, doc_tokens):
        nq = len(qry_tokens)
        nd = max(1, len(doc_tokens))
        doc_length_surplus = max(1, nd / nq)
        doc_length_penalty_factor = 1 + 0.5 * math.sqrt(doc_length_surplus)
        return doc_length_penalty_factor
