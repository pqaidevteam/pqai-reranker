import unittest
import numpy as np

import sys
from pathlib import Path
from dotenv import load_dotenv

test_dir = str(Path(__file__).parent.resolve())
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR.resolve()))
load_dotenv(f"{BASE_DIR}/.env")

from core.reranker import Ranker
from core.reranker import CustomRanker
from core.reranker import ConceptMatchRanker

query = "This is a red apple, which is a fruit"

documents = [
    "This is a red car",
    "This is a green apple",
    "There are many red coloured fruits, apple is one of them",
    "An apple a day, keeps the doctor away",
    "There is a lion in the forest",
]


class TestReRankerClass(unittest.TestCase):
    def setUp(self):
        self.dummy_reranker_fn = lambda str1, str2: len(str2.split())

    def test_score(self):
        reranker = Ranker(self.dummy_reranker_fn)
        query = "This is a red apple, which is a fruit"
        document = "This is a green apple"
        expected = len(document.split())
        actual = reranker.score(query, document)
        self.assertEqual(expected, actual)

    def test_rank(self):
        reranker = Ranker(self.dummy_reranker_fn)
        expected = [len(doc.split()) for doc in documents]
        expected = np.argsort(expected)[::-1]
        actual = reranker.rank(query, documents)
        self.assertEqual(0, len(set(expected) - set(actual)))


class TestCustomRanker(unittest.TestCase):
    def test_score(self):
        reranker = CustomRanker()
        query = "This is a red apple, which is a fruit"
        document = "This is a green apple"
        actual = reranker.score(query, document)
        self.assertIsInstance(actual, float)

    def test_rank(self):
        reranker = CustomRanker()
        actual = reranker.rank(query, documents)
        self.assertIsInstance(actual, np.ndarray)
        self.assertGreater(len(actual), 0)


class TestConceptMatchRanker(unittest.TestCase):
    def test_score(self):
        reranker = ConceptMatchRanker()
        query = "This is a red apple, which is a fruit"
        document = "This is a green apple"
        actual = reranker.score(query, document)
        self.assertIsInstance(actual, float)

    def test_rank(self):
        reranker = ConceptMatchRanker()
        actual = reranker.rank(query, documents)
        self.assertIsInstance(actual, np.ndarray)
        self.assertGreater(len(actual), 0)


if __name__ == "__main__":
    unittest.main()
