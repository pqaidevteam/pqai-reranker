"""Test for service HTTP API
"""
import sys
import unittest
from pathlib import Path
from dotenv import load_dotenv
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).parent.parent.resolve()
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH.as_posix())
sys.path.append(BASE_DIR.as_posix())

from main import app
from test_utils import QUERY, DOCUMENTS

MODELS = ["concept-match-ranker", "custom-ranker"]


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test__rerank_route(self):
        for model in MODELS:
            data = {"query": QUERY, "docs": DOCUMENTS, "model": model}
            response = self.client.post("/rerank", json=data)
            self.assertEqual(200, response.status_code)
            ranks = response.json().get("ranks")
            self.assertIsInstance(ranks, list)
            self.assertEqual(len(DOCUMENTS), len(ranks))

    def test__returns_error_for_invalid_model_name(self):
        model = "a-model-that-doesnt-exist"
        data = {"query": QUERY, "docs": DOCUMENTS, "model": model}
        response = self.client.post("/rerank", json=data)
        self.assertEqual(400, response.status_code)

    def test__returns_error_for_empty_doc_list(self):
        model = "concept-match-ranker"
        data = {"query": QUERY, "docs": [], "model": model}
        response = self.client.post("/rerank", json=data)
        self.assertEqual(400, response.status_code)

    def test__score_route(self):
        for model in MODELS:
            data = {"query": QUERY, "doc": DOCUMENTS[1], "model": model}
            response = self.client.post("/score", json=data)
            self.assertEqual(200, response.status_code)
            score = response.json().get("score")
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
