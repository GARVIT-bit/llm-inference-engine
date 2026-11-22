"""Evaluation data: WikiText-2 (raw) test split, fetched as parquet from the HF Hub."""
from __future__ import annotations

from pathlib import Path

from engine.weights import WEIGHTS_DIR

DATA_DIR = WEIGHTS_DIR / "datasets" / "wikitext"
_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def wikitext2_test_text() -> str:
    """The test split joined the standard way ("\\n\\n".join of the rows, as in HF's ppl guide)."""
    import pyarrow.parquet as pq

    path = DATA_DIR / _FILE
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("Salesforce/wikitext", _FILE, repo_type="dataset",
                                    local_dir=str(DATA_DIR)))
    return "\n\n".join(pq.read_table(path).column("text").to_pylist())
