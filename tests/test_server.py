"""Test for service HTTP API

Attributes:
    ENV_FILE (str): Absolute path to .env file (used for reading port no.)
    HOST (str): IP address of the host where service is running
    PORT (str): Port no. on which the server is listening
    PROTOCOL (str): `http` or `https`
"""
import os
import unittest
import socket
from pathlib import Path
import requests
from dotenv import load_dotenv

ENV_FILE = str((Path(__file__).parent.parent / ".env").resolve())
load_dotenv(ENV_FILE)

# pylint: disable=wrong-import-position

from test_utils import QUERY
from test_utils import DOCUMENTS

PROTOCOL = "http"
HOST = "localhost"
PORT = os.environ["PORT"]
API_ENDPOINT = "{PROTOCOL}://{HOST}:{PORT}"

MODELS = ["concept-match-ranker", "custom-ranker"]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_not_running = sock.connect_ex((HOST, int(PORT))) != 0

if server_not_running:
    print("Server is not running. API tests will be skipped.")


# pylint: disable=missing-function-docstring, missing-class-docstring

@unittest.skipIf(server_not_running, "Works only when true")
class TestAPI(unittest.TestCase):

    def test__rerank_route(self):
        for model in MODELS:
            data = {"query": QUERY, "docs": DOCUMENTS, "model": model}
            response = self.call_route("/rerank", data, "post")
            self.assertEqual(200, response.status_code)
            ranks = response.json().get("ranks")
            self.assertIsInstance(ranks, list)
            self.assertEqual(len(DOCUMENTS), len(ranks))

    def test__returns_error_for_invalid_model_name(self):
        model = "a-model-that-doesnt-exist"
        data = {"query": QUERY, "docs": DOCUMENTS, "model": model}
        response = self.call_route("/rerank", data, "post")
        self.assertEqual(400, response.status_code)

    def test__returns_error_for_empty_doc_list(self):
        model = "concept-match-ranker"
        data = {"query": QUERY, "docs": [], "model": model}
        response = self.call_route("/rerank", data, "post")
        self.assertEqual(400, response.status_code)

    def test__score_route(self):
        for model in MODELS:
            data = {"query": QUERY, "doc": DOCUMENTS[1], "model": model}
            response = self.call_route("/score", data, "post")
            self.assertEqual(200, response.status_code)
            score = response.json().get("score")
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)

    @staticmethod
    def call_route(route, data, method="get"):
        route = route.lstrip("/")
        url = f"{PROTOCOL}://{HOST}:{PORT}/{route}"
        response = getattr(requests, method)(url, json=data)
        return response


if __name__ == "__main__":
    unittest.main()
