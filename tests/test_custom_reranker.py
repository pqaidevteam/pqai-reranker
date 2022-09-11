"""
Tests for Custom Reranker
"""

import unittest
import sys
import numpy as np

from test_utils import init_env, BASE_DIR, QUERY, DOCUMENTS

init_env()
sys.path.append(BASE_DIR)

# pylint: disable=wrong-import-position
from core.custom_reranker import (
    CustomRanker,
    GloveWordEmbeddings,
    VectorSequence,
    Interaction,
    InteractionMatrix,
)

# pylint: disable=missing-function-docstring,missing-class-docstring

class TestCustomRanker(unittest.TestCase):

    def test_score(self):
        reranker = CustomRanker()
        query = "This is a red apple, which is a fruit"
        document = "This is a green apple"
        actual = reranker.score(query, document)
        self.assertIsInstance(actual, float)

    def test_rank(self):
        reranker = CustomRanker()
        actual = reranker.rank(QUERY, DOCUMENTS)
        self.assertIsInstance(actual, np.ndarray)
        self.assertGreater(len(actual), 0)


class TestGloveWordEmbeddings(unittest.TestCase):

    def setUp(self):
        self.We = GloveWordEmbeddings()

    def test__can_return_vocab_size(self):
        n = len(self.We)
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test__can_return_vector_by_word_index(self):
        we = self.We[0]  # we = word embedding
        self.assertIsInstance(we, np.ndarray)
        self.assertEqual(we.shape, (256,))

    def test__can_return_vector_by_word(self):
        we = self.We["the"]
        self.assertIsInstance(we, np.ndarray)
        self.assertEqual(we.shape, (256,))

    def test__returns_zero_vector_for_unk(self):
        unk = "django"
        we = self.We[unk]
        self.assertIsInstance(we, np.ndarray)
        self.assertEqual(we.shape, (256,))
        self.assertEqual(0, sum(we))

    def test__can_return_smooth_inverse_frequency(self):
        sif_the = self.We.get_sif("the")
        sif_invention = self.We.get_sif("invention")
        self.assertIsInstance(sif_the, float)
        self.assertIsInstance(sif_invention, float)
        self.assertGreater(sif_invention, sif_the)

    def test__can_return_smooth_inverse_frequency_of_unk(self):
        sif_unk = self.We.get_sif("django")
        self.assertEqual(1.0, sif_unk)


class TestVectorSequence(unittest.TestCase):

    def setUp(self):
        We = GloveWordEmbeddings()
        self.tokens = ["a", "fire", "fighting", "drone"]
        self.vectors = [We[token] for token in self.tokens]
        self.seq = VectorSequence(self.tokens, self.vectors)

    def test__returns_str_repr(self):
        string_repr = str(self.seq)
        self.assertIsInstance(string_repr, str)

    def test__can_weigh_vectors_by_arr(self):
        weight_arr = [0.1, 0.8, 0.8, 0.6]
        self.assertWeighingChangesSequence(weight_arr)

    def test__can_weigh_vectors_by_dict(self):
        weight_dict = {"a": 0.1, "fire": 0.8, "fighting": 0.8, "drone": 0.6}
        self.assertWeighingChangesSequence(weight_dict)

    def test__can_weigh_vectors_by_np_array(self):
        weights = np.array([0.1, 0.8, 0.8, 0.6])
        self.assertWeighingChangesSequence(weights)

    def assertWeighingChangesSequence(self, weights):
        M0 = np.copy(self.seq.matrix)
        self.seq.weigh(weights)
        M1 = np.copy(self.seq.matrix)
        self.assertIsInstance(M0, np.ndarray)
        self.assertIsInstance(M1, np.ndarray)
        self.assertEqual(M0.shape, M1.shape)
        self.assertNotEqual(np.sum(M1 - M0), 0.0)

    def test__can_enforce_fixed_length_more_than_seq_length(self):
        self.seq.set_length(20)
        M = np.copy(self.seq.matrix)
        self.assertEqual(M.shape, (20, 256))

    def test__can_enforce_fixed_length_smaller_than_seq_length(self):
        self.seq.set_length(3)
        M = np.copy(self.seq.matrix)
        self.assertEqual(M.shape, (3, 256))

    def test__can_return_normalized_matrix(self):
        Mn = self.seq.normalized_matrix
        for vector in Mn:
            norm = np.linalg.norm(vector)
            self.assertEqual(norm, 1.0)


class TestInteraction(unittest.TestCase):

    def setUp(self):
        metrics = ["cosine", "dot", "euclidean"]
        context = [True, False]
        amplify = [True, False]
        reinforce = [True, False]
        self.interactions = [
            Interaction(metric=m, context=c, amplify=a, reinforce=r)
            for m in metrics
            for c in context
            for a in amplify
            for r in reinforce
        ]

    def test__can_create_interaction_matrix(self):
        We = GloveWordEmbeddings()

        labels_a = ["a", "fire", "fighting", "drone"]
        vectors_a = [We[token] for token in labels_a]
        seq_a = VectorSequence(labels_a, vectors_a)

        labels_b = ["a", "power", "line", "aerial", "vehicle"]
        vectors_b = [We[token] for token in labels_b]
        seq_b = VectorSequence(labels_b, vectors_b)

        for interaction in self.interactions:
            self.assertIsInstance(interaction.interact(seq_a, seq_b), InteractionMatrix)


if __name__ == "__main__":
    unittest.main()
