"""Test for service HTTP API

Attributes:
    ENV_FILE (str): Absolute path to .env file (used for reading port no.)
    HOST (str): IP address of the host where service is running
    PORT (str): Port no. on which the server is listening
    PROTOCOL (str): `http` or `https`
"""
import os
import unittest
from pathlib import Path
import requests
from dotenv import load_dotenv

ENV_FILE = str((Path(__file__).parent.parent / ".env").resolve())
load_dotenv(ENV_FILE)

PROTOCOL = "http"
HOST = "localhost"
PORT = os.environ["PORT"]
API_ENDPOINT = "{}://{}:{}".format(PROTOCOL, HOST, PORT)

MODELS = ["concept-match-ranker", "custom-ranker"]


class TestAPI(unittest.TestCase):
    """Test API routes"""

    def setUp(self):
        self.query = "This is a red apple, which is a fruit"
        self.docs = [
            "This is a red car",
            "This is a green apple",
            "There are many red coloured fruits, apple is one of them",
            "An apple a day, keeps the doctor away",
            "There is a lion in the forest",
        ]

    def test__rerank_route(self):
        """Check if a valid request returns a valid response"""
        for model in MODELS:
            data = {"query": self.query, "docs": self.docs, "model": model}
            response = self.call_route("/rerank", data, "post")
            self.assertEqual(200, response.status_code)
            ranks = response.json().get("ranks")
            self.assertIsInstance(ranks, list)
            self.assertEqual(len(self.docs), len(ranks))

    def test__returns_error_for_invalid_model_name(self):
        """Check if getting appropriate response code for request parameters"""
        model = "a-model-that-doesnt-exist"
        data = {"query": self.query, "docs": self.docs, "model": model}
        response = self.call_route("/rerank", data, "post")
        self.assertEqual(400, response.status_code)

    def test__returns_error_for_empty_doc_list(self):
        """Check if getting appropriate response code for request parameters"""
        model = "concept-match-ranker"
        data = {"query": self.query, "docs": [], "model": model}
        response = self.call_route("/rerank", data, "post")
        self.assertEqual(400, response.status_code)

    def test__score_route(self):
        """Check if a valid request return a valid response"""
        for model in MODELS:
            data = {"query": self.query, "doc": self.docs[1], "model": model}
            response = self.call_route("/score", data, "post")
            self.assertEqual(200, response.status_code)
            score = response.json().get("score")
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)

    def call_route(self, route, data, method="get"):
        """Make a request to given route with given parameters"""
        route = route.lstrip("/")
        url = f"{PROTOCOL}://{HOST}:{PORT}/{route}"
        response = getattr(requests, method)(url, json=data)
        return response


if __name__ == "__main__":
    unittest.main()
