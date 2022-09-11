"""
Tests for Rerankers
"""

import unittest
import sys
import numpy as np

from test_utils import BASE_DIR, QUERY, DOCUMENTS, init_env

init_env()
sys.path.append(BASE_DIR)

# pylint: disable=wrong-import-position

from core.reranker import Ranker, ConceptMatchRanker

# pylint: disable=missing-function-docstring, missing-class-docstring

class DummyRanker(Ranker):

    def __init__(self, score_type):
        super().__init__(score_type)

    def score(self, query, document):
        return len(document.split()) - len(query.split())


class TestReRankerClass(unittest.TestCase):

    def test_score(self):
        reranker = DummyRanker("distance")
        expected = -4
        actual = reranker.score(QUERY, DOCUMENTS[1])
        self.assertEqual(expected, actual)

    def test_rank(self):
        reranker = DummyRanker("distance")
        expected = [len(doc.split()) for doc in DOCUMENTS]
        expected = np.argsort(expected)[::-1]
        actual = reranker.rank(QUERY, DOCUMENTS)
        self.assertEqual(0, len(set(expected) - set(actual)))


class TestConceptMatchRanker(unittest.TestCase):

    def test_score(self):
        reranker = ConceptMatchRanker()
        actual = reranker.score(QUERY, DOCUMENTS[1])
        self.assertIsInstance(actual, float)

    def test_rank(self):
        reranker = ConceptMatchRanker()
        actual = reranker.rank(QUERY, DOCUMENTS)
        self.assertIsInstance(actual, np.ndarray)
        self.assertGreater(len(actual), 0)


if __name__ == "__main__":
    unittest.main()
