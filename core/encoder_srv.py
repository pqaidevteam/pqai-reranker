"""Encoder service"""

import os
import requests

ENCODER_SRV_ENDPOINT = os.environ["ENCODER_SRV_ENDPOINT"]


def encode(data, encoder):
    """Encode given data using given encoder"""
    payload = {"data": data, "encoder": encoder}
    try:
        response = requests.post(f"{ENCODER_SRV_ENDPOINT}/encode", json=payload)
    except Exception as e:
        raise e
    if response.status_code != 200:
        raise Exception(response.text)
    return response.json().get("encoded")
