"""
Matchpyramid reranker

Pang, Liang, et al. "Text matching as image recognition."
Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 30. No. 1. 2016.
"""

import types
from pathlib import Path
import numpy as np
from matchzoo.preprocessors import BasicPreprocessor
from matchzoo.engine.base_preprocessor import load_preprocessor
from matchzoo.engine.base_model import load_model
from matchzoo.preprocessors.chain_transform import chain_transform

BASE_DIR = str(Path(__file__).parent.parent.resolve())
ASSETS_DIR = f"{BASE_DIR}/assets"

MODEL_DIR = f"{ASSETS_DIR}/MatchPyramid_200_tokens"
model_preprocessor = load_preprocessor(f"{MODEL_DIR}/preprocessor")
model = load_model(f"{MODEL_DIR}/model")

MAX_LEN = 200
"""
	Set embeddings for the <pad> and <oov> terms
	to zero vectors so that they have zero interaction
	among themselves and with other terms.
"""
M = model.get_embedding_layer().get_weights()[0]
M[0] = np.zeros(256)
M[1] = np.zeros(256)
model.get_embedding_layer().set_weights([M])


def get_d_pool_array(n_docs, max_len):
    """Get document pool array"""
    d_pool_list = []
    for i in range(max_len):
        temp_list = []
        for j in range(max_len):
            temp_list.append([i, j])
        d_pool_list.append(temp_list)

    arr = []
    for i in range(n_docs):
        arr.append(d_pool_list)
    d_pool_array = np.asarray(arr)

    del d_pool_list
    del arr

    return d_pool_array

# pylint: disable=protected-access

def get_transformer(preprocessor: BasicPreprocessor, mode: str) -> types.FunctionType:
    """Get transformation function"""
    transformer_units = preprocessor._units[:]
    if mode == "right":
        transformer_units.append(preprocessor._context["filter_unit"])

    transformer_units.append(preprocessor._context["vocab_unit"])
    if mode == "right":
        transformer_units.append(preprocessor._right_fixedlength_unit)
    else:
        transformer_units.append(preprocessor._left_fixedlength_unit)
    transformer = chain_transform(transformer_units)
    return transformer


transformer_left = get_transformer(model_preprocessor, "left")
transformer_right = get_transformer(model_preprocessor, "right")


def get_similarity_scores(texts_left, texts_right):
    """Get similarity score between two texts (e.g. query and document)"""
    inputs_left = [transformer_left(text) for text in texts_left]
    inputs_right = [transformer_right(text) for text in texts_right]
    output = model.predict([inputs_left, inputs_right, get_d_pool_array(1, MAX_LEN)])[0]
    return output


def calculate_similarity(left_val, right_val):
    """Get similarity score between two or more texts (e.g. a query and many documents)"""
    assert isinstance(left_val, (str, list))
    assert isinstance(right_val, (str, list))

    if isinstance(left_val, str) and isinstance(right_val, str):
        return get_similarity_scores([left_val], [right_val])[0]
    if isinstance(left_val, str) and isinstance(right_val, list):
        n = len(right_val)
        return [calculate_similarity(a, b) for a, b in zip(n*[left_val], right_val)]
    if isinstance(left_val, list) and isinstance(right_val, list):
        assert len(left_val) == len(right_val)
        return [calculate_similarity(a, b) for a, b in zip(left_val, right_val)]
    raise TypeError("Invalid arguments passed to `calculate_similarity` function")
