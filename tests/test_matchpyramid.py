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
        pass

    def test__normal_operation(self):
        text1 = "This invention relates with coffee makers."
        text2 = "A coffee making machine has been disclosed."
        sim = calculate_similarity(text1, text2)
        self.assertIsInstance(sim, np.float32)

    def test__can_match_string_with_itself(self):
        text1 = "This invention relates with coffee makers."
        text2 = "A coffee making machine has been disclosed."
        sim0 = calculate_similarity(text1, text1)
        sim1 = calculate_similarity(text1, text2)
        self.assertGreater(sim0, sim1)


if __name__ == "__main__":
    unittest.main()
