"""
Shared items used by tests
"""
from pathlib import Path
from dotenv import load_dotenv

TEST_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())

def init_env():
    """Load environment variables from .env file
    """
    filepath = f"{BASE_DIR}/.env"
    load_dotenv(filepath)

QUERY = "This is a red apple, which is a fruit"
DOCUMENTS = [
    "This is a red car",
    "This is a green apple",
    "There are many red coloured fruits, apple is one of them",
    "An apple a day, keeps the doctor away",
    "There is a lion in the forest",
]
