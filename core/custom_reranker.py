"""Summary

Attributes:
    BASE_DIR (TYPE): Description
    embeddings (TYPE): Description
    MODELS_DIR (TYPE): Description
    sifs (TYPE): Description
"""
import math
from pathlib import Path
import json
import numba
import numpy as np
from nltk.tokenize import RegexpTokenizer

from scipy.spatial import distance
from core.encoder_srv import encode
from core.reranker import Ranker

BASE_DIR = str(Path(__file__).parent.parent.resolve())
MODELS_DIR = "{}/assets/".format(BASE_DIR)

class GloveWordEmbeddings:

    """Glove word embeddings"""

    def __init__(self):
        """Initialize"""
        self._models_dir = MODELS_DIR
        self._vocab_file = f"{self._models_dir}/glove-vocab.json"
        self._dict_file = f"{self._models_dir}/glove-dictionary.json"
        self._dfs_file = f"{self._models_dir}/dfs.json"
        self._embs_file = f"{self._models_dir}/glove-We.npy"
        self._vocab = None
        self._dictionary = None
        self._dfs = None
        self._sifs = None
        self._embs = None
        self._dims = None
        self._load()

    def _load(self):
        """Load the embeddings data from the disk
        """
        with open(self._vocab_file) as file:
            self._vocab = json.load(file)
        with open(self._dict_file) as file:
            self._dictionary = json.load(file)
        with open(self._dfs_file) as file:
            self._dfs = json.load(file)
        self._embs = np.load(self._embs_file)
        self._sifs = {word: self.df2sif(word, self._dfs) for word in self._dfs}
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
        if type(item) is int:
            return self._embs[item]
        elif type(item) is str:
            item = item if item in self._dictionary else "<unk>"
            return self._embs[self._dictionary[item]]
        else:
            return np.zeros(self._dims)

    def get_sif(self, word):
        """Return SIF for a given word

        Args:
            word (str): Word

        Returns:
            float: SIF value
        """
        return self._sifs.get(word, 1.0)


embeddings = GloveWordEmbeddings()
sifs = embeddings._sifs


class TokenSequence(list):

    """Summary
    """

    def __init__(self, tokens):
        """Summary

        Args:
            tokens (TYPE): Description
        """
        super().__init__(tokens)
        self._tokens = tokens

    def to_vector_sequence(self, token_embeddings):
        """Summary

        Args:
            token_embeddings (TYPE): Description

        Returns:
            TYPE: Description
        """
        vectors = [token_embeddings[token] for token in self._tokens]
        return VectorSequence(self._tokens, vectors)

    @property
    def tokens(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self._tokens


class VectorSequence:

    """Summary
    """

    def __init__(self, labels, vectors):
        """Summary

        Args:
            labels (TYPE): Description
            vectors (TYPE): Description
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
        """Summary

        Returns:
            TYPE: Description
        """
        return self._labels

    def __repr__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        text = f"VectorSequence: {len(self._labels)} labels, {len(self._sequence)} vectors;"
        text += f' Labels: {", ".join(self._labels[:5])}'
        text += ", ..." if self._labels[5:] else ""
        return text

    def _weighted_by_tokens(self, weights):
        """Summary

        Args:
            weights (TYPE): Description

        Returns:
            TYPE: Description
        """
        W = [weights[token] for token in self._tokens]
        return self.weighted_by_vectors(W)

    def _weighted_by_vectors(self, W):
        """Summary

        Args:
            W (TYPE): Description

        Returns:
            TYPE: Description
        """
        W = np.array(W).reshape(1, -1)
        return self._sequence * W.T

    def weigh(self, weights):
        """Summary

        Args:
            weights (TYPE): Description
        """
        if isinstance(weights, dict):
            self._weighted_by_tokens(weights)
        self._weighted_by_vector(weights)

    @property
    def redundancy_vector(self):
        """Summary

        Returns:
            TYPE: Description
        """
        interact = self._default_interaction.interact
        interactions = interact(self, self)
        interactions = np.tril(interactions._matrix, -1)
        return np.max(interactions, axis=1)

    @property
    def matrix(self):
        """Summary

        Returns:
            TYPE: Description
        """
        if self._fixed_length is None:
            return self._sequence
        if self._n > self._fixed_length:
            return self._truncated
        else:
            return self._padded

    @property
    def _truncated(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self._sequence[: self._fixed_length]

    @property
    def _padded(self):
        """Summary

        Returns:
            TYPE: Description
        """
        r = self._fixed_length - self._n
        shape = (r, self._dims)
        padding = np.zeros(shape)
        return np.concatenate((self._sequence, padding))

    def set_length(self, n):
        """Summary

        Args:
            n (TYPE): Description

        Returns:
            TYPE: Description
        """
        self._fixed_length = n
        return self

    @property
    def normalized_matrix(self):
        """Summary

        Returns:
            TYPE: Description
        """
        row_magnitudes = np.sqrt(
            np.sum(self._sequence * self._sequence, axis=1, keepdims=True)
        )
        row_magnitudes += np.finfo(float).eps
        return self._sequence / row_magnitudes


class Interaction:

    """Summary

    Attributes:
        amplify (TYPE): Description
        context (TYPE): Description
        metric (TYPE): Description
        reinforce (TYPE): Description
        window_size (TYPE): Description
    """

    def __init__(self,
                 metric="cosine",
                 context=False,
                 amplify=False,
                 reinforce=False,
                 window=5):
        """Summary

        Args:
            metric (str, optional): Description
            context (bool, optional): Description
            amplify (bool, optional): Description
            reinforce (bool, optional): Description
            window (int, optional): Description
        """
        self.metric = metric
        self.context = context
        self.amplify = amplify
        self.reinforce = reinforce
        self.window_size = window
        self._amplify_matrix = np.vectorize(self._amplify)
        self._a = 3.2
        self._b = 7.5
        self._c = 0.46
        self._f = 1.0
        self._h = 0.0

    def _dot_interaction(self, A, B):
        """Summary

        Args:
            A (TYPE): Description
            B (TYPE): Description

        Returns:
            TYPE: Description
        """
        return np.matmul(A, B.T)

    def _cosine_interaction(self, A, B):
        """Summary

        Args:
            A (TYPE): Description
            B (TYPE): Description

        Returns:
            TYPE: Description
        """
        An = self._normalize_rows(A)
        Bn = self._normalize_rows(B)
        return self._dot_interaction(An, Bn)

    def _euclidean_interaction(self, A, B):
        """Summary

        Args:
            A (TYPE): Description
            B (TYPE): Description

        Returns:
            TYPE: Description
        """
        diff = A - B
        sq_diff = diff * diff
        return np.sqrt(sq_diff)

    def _context_sequence(self, vector_seq):
        """Summary

        Args:
            vector_seq (TYPE): Description

        Returns:
            TYPE: Description
        """
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
        """Summary

        Args:
            vector_seq_A (TYPE): Description
            vector_seq_B (TYPE): Description

        Returns:
            TYPE: Description
        """
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
        """Summary

        Returns:
            TYPE: Description
        """
        if self.metric == "cosine":
            return self._cosine_interaction
        elif self.metric == "dot":
            return self._dot_interaction
        elif self.metric == "euclidean":
            return self._euclidean_interaction

    @staticmethod
    def _normalize_rows(M):
        """Summary

        Args:
            M (TYPE): Description

        Returns:
            TYPE: Description
        """
        row_magnitudes = np.sqrt(np.sum(M * M, axis=1, keepdims=True))
        row_magnitudes += np.finfo(float).eps
        return M / row_magnitudes

    @staticmethod
    def _reinforce(A, B):
        """Summary

        Args:
            A (TYPE): Description
            B (TYPE): Description

        Returns:
            TYPE: Description
        """
        return 0.25 * (A + B + 2 * (A * B))

    def _amplify(self, x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        return self._h + (self._f / (1 + (self._a * math.exp(self._b * (x - self._c)))))

    @staticmethod
    @numba.vectorize([numba.float64(numba.float64)])
    def _amplifier(x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        return 1 / (1 + (3.2 * math.exp(-7.5 * (x - 0.46))))


class InteractionMatrix:

    """Summary
    """

    def __init__(self, I):
        """Summary

        Args:
            I (TYPE): Description
        """
        self._matrix = I

    def available_metrics(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self._available_interactions

    def maxpool(self, direction="horizontal"):
        """Summary

        Args:
            direction (str, optional): Description

        Returns:
            TYPE: Description
        """
        axis = 1 if direction == "horizontal" else 0
        return np.max(self._matrix, axis=axis)

class CustomRanker(Ranker):

    """Summary
    """

    def __init__(self):
        """Summary
        """
        self._interaction = Interaction()
        self._interaction.metric = "cosine"
        self._interaction.amplify = True
        self._interaction.reinforce = False
        self._interaction.context = False
        self._interaction.window = 10
        self._interact = self._interaction.interact
        super().__init__("similarity")

    def score(self, query, doc):
        """Summary

        Args:
            query (str): Query
            doc (str): Document

        Returns:
            float: Similarity score between query and document
        """
        query_tokens = TokenSequence(RegexpTokenizer(r"\w+").tokenize(query.lower()))
        doc_tokens = TokenSequence(RegexpTokenizer(r"\w+").tokenize(doc.lower()))
        nq = len(query_tokens)
        nd = max(1, len(doc_tokens))
        doc_length_surplus = max(1, nd / nq)
        doc_length_penalty_factor = 1 + 0.5 * math.sqrt(doc_length_surplus)
        Q = query_tokens.to_vector_sequence(embeddings)
        D = doc_tokens.to_vector_sequence(embeddings)
        query_term_matches = self._interact(Q, D).maxpool()
        sifs = embeddings._sifs
        query_term_weights = [
            (sifs[word] if word in sifs else 1.0) for word in query_tokens
        ]
        query_term_weights *= 1 - Q.redundancy_vector
        query_term_matches *= query_term_weights
        score = query_term_matches.sum()
        if query != doc:
            score /= self.score(query, query)
            score /= doc_length_penalty_factor
        return score
