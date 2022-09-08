import unittest
from unittest import TestCase
import sys
import re
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR.resolve()))
load_dotenv(f"{BASE_DIR}/.env")

from core.matchpyramid import calculate_similarity


class TestMatchPyramidRanker(TestCase):

    def setUp(self):
        self.text1 = "This invention relates with coffee makers."
        self.text2 = "A coffee making machine has been disclosed."

    def test__can_match_query_with_document(self):
        
        sim = calculate_similarity(self.text1, self.text2)
        self.assertIsInstance(sim, np.float32)

    def test__can_match_string_with_itself(self):
        sim0 = calculate_similarity(self.text1, self.text1)
        sim1 = calculate_similarity(self.text1, self.text2)
        self.assertGreater(sim0, sim1)

    def test__can_match_query_with_many_documents(self):
        sims = calculate_similarity(self.text1, [self.text1, self.text2])
        self.assertIsInstance(sims, list)
        self.assertEqual(2, len(sims))
        for score in sims:
            self.assertIsInstance(score, np.float32)

    def test__can_match_queries_with_many_documents(self):
        arr = [self.text1, self.text2]
        sims = calculate_similarity(arr, arr)
        self.assertIsInstance(sims, list)
        self.assertEqual(2, len(sims))
        for score in sims:
            self.assertIsInstance(score, np.float32)


if __name__ == "__main__":
    unittest.main()
