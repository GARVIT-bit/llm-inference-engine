import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def gpt2_small():
    from engine.weights import load_gpt2
    return load_gpt2("gpt2")
