"""
Reranking models: they take in a set of documents and ranks them accoring to
their similarity to a query.
"""

import math
import numpy as np
from scipy.spatial import distance
from core.encoder_srv import encode


class Ranker:

    """Summary

    Attributes:
        metric_type (TYPE): Description
    """

    def __init__(self, metric_type="similarity"):
        """Summary

        Args:
            metric_type (str, optional): Description
        """
        self.metric_type = metric_type

    def score(self, query, document):
        """Calculate numerical similarity between query and document."""
        raise NotImplementedError

    def rank(self, query, documents):
        """Get ranks for `documents` on the basis of similarity with
            `query`.

        Args:
            query (str): The query (reference text)
            documents (list): Text documents

        Returns:
            list: Ranks for each of the documents, e.g., [2, 0, 1] means
                the document at index 0 in the input list `documents` has
                rank 2 (least similar) and document at index 1 is most
                similar. Note that the calling function has to sort the
                actual document list.
        """
        scores = [self.score(query, document) for document in documents]
        ranks = np.argsort(scores)
        if self.metric_type == "similarity":
            ranks = ranks[::-1]
        return ranks


class ConceptMatchRanker(Ranker):

    """Ranking algorithm that scores text similarity by extracting concepts
    (entities) from a piece of text, then comparing their embeddings
    with word movers distance.
    """

    def __init__(self):
        """Initialize
        """
        super().__init__("distance")

    def score(self, query, document):
        """Get a numerical similarity between given query and doc.

        Args:
            query (str): Query
            document (str): Document

        Returns:
            float: Distance between query and document
        """
        query_rep = encode(encode(query, "boe"), "emb")
        document_rep = encode(encode(document, "boe"), "emb")
        return ConceptMatchRanker._wmd(query_rep, document_rep)

    @staticmethod
    def _wmd(bov1, bov2, dist_fn=distance.cosine):
        """Calculate word movers distance

        Args:
            bov1 (List[List[float]]): Bag of vectors 1
            bov2 (List[List[float]]): Bag of vectors 2
            dist_fn (method, optional): Distance function

        Returns:
            float: Word movers distance
        """
        n1 = len(bov1)
        n2 = len(bov2)
        if n1 == 0 or n2 == 0:
            return math.inf
        dists = np.zeros((n1, n2))
        for i, v1 in enumerate(bov1):
            for j, v2 in enumerate(bov2):
                dists[i, j] = dist_fn(v1, v2)
        return dists.min(axis=1).sum()
