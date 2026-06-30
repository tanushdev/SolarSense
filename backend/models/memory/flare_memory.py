"""
Historical Flare Memory Bank
============================
Retrieves the most similar historical flare events for any current observation.

WHY THIS IS POWERFUL:
  A space physicist looking at a pre-flare signature doesn't just use a model —
  they recall past events. "This looks like the 2003 Halloween storm precursor."
  
  This module gives the AI system the same capability.
  When the forecaster issues an alert, the operator can see:
    "Alert: M-class flare likely | Similar to events on 2023-08-05 (94% similar),
     2024-02-18 (89% similar) — both preceded M-class events in 8–12 minutes."
  
  This is explainability through analogy — more interpretable than attention maps
  for operational space-weather personnel.

IMPLEMENTATION:
  1. Embed each historical flare event as a feature vector
     (physics features averaged over the 30-min pre-flare window)
  2. Store embeddings in a FAISS flat index
  3. At inference, embed the current 30-min window and query top-k neighbors
  4. Return similar events with metadata

BUILD STEP: Run scripts/build_memory_bank.py once on the historical catalog.
"""

import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from typing import List


@dataclass
class SimilarEvent:
    """A retrieved similar historical flare."""
    event_id:    str
    date:        str
    flare_class: str
    similarity:  float     # 0–1, higher = more similar
    lead_time:   float     # Minutes between this retrieval point and peak
    description: str       # Human-readable summary


class FlareMemoryBank:
    """
    FAISS-backed historical flare event memory bank.
    """

    def __init__(self, index_path: str = "models/checkpoints/faiss_memory.index",
                 metadata_path: str = "models/checkpoints/faiss_metadata.pkl",
                 embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.index_path    = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index         = None
        self.metadata      = None

    def build(self, event_embeddings: np.ndarray,
              event_metadata: List[dict]):
        """
        Build FAISS index from historical event embeddings.
        
        Parameters
        ----------
        event_embeddings : (N, embedding_dim) float32 array
        event_metadata   : list of dicts with event info
        """
        assert event_embeddings.dtype == np.float32
        assert event_embeddings.shape[1] == self.embedding_dim

        # Normalize for cosine similarity
        faiss.normalize_L2(event_embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine after L2 norm
        self.index.add(event_embeddings)
        self.metadata = event_metadata

        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("FAISS memory bank built: {} events indexed", len(event_metadata))

    def load(self):
        """Load pre-built index from disk."""
        self.index    = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info("FAISS memory bank loaded: {} events", self.index.ntotal)

    def query(self, query_embedding: np.ndarray,
              top_k: int = 5) -> List[SimilarEvent]:
        """
        Find top-k most similar historical events.
        
        Parameters
        ----------
        query_embedding : (1, embedding_dim) float32
        
        Returns
        -------
        List[SimilarEvent] sorted by similarity descending
        """
        if self.index is None:
            raise RuntimeError("Memory bank not loaded. Call load() first.")
        q = query_embedding.astype(np.float32)
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            results.append(SimilarEvent(
                event_id=meta.get("event_id", str(idx)),
                date=meta.get("date", "unknown"),
                flare_class=meta.get("flare_class", "?"),
                similarity=float(dist),
                lead_time=float(meta.get("lead_time_minutes", -1)),
                description=meta.get("description", ""),
            ))
        return results