from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

_STORE_PATH = Path(__file__).resolve().parent.parent / "gallery.npz"


def load(path: Path = _STORE_PATH) -> dict[str, npt.NDArray[np.float32]]:
    """Loads the face store (user_id -> embedding) from disk; empty if none.

    Args:
        path: Path to the .npz file containing the face store.

    Returns:
        A dictionary mapping user_id (str) to embedding (np.ndarray of shape (512,)).
    """
    if not path.exists():
        return {}
    with np.load(path) as data:
        labels: npt.NDArray[np.str_] = data["labels"]
        embeddings: npt.NDArray[np.float32] = data["embeddings"]
    return {
        str(labels[i]): np.asarray(embeddings[i], dtype=np.float32)
        for i in range(len(labels))
    }


def save(store: Mapping[str, npt.NDArray[np.float32]], path: Path = _STORE_PATH) -> None:
    """Saves the face store (user_id -> embedding) to disk.

    Args:
        store: A dictionary mapping user_id (str) to embedding (np.ndarray of shape (512,)).
        path: Path to the .npz file where the face store will be saved.
    """
    names = list(store)
    np.savez(
        path,
        labels=np.array(names),
        embeddings=np.stack([store[name] for name in names]),
    )


def match(
    store: Mapping[str, npt.NDArray[np.float32]],
    embedding: npt.NDArray[np.float32],
    threshold: float = 0.3,
) -> tuple[str, float]:
    """Finds the closest enrolled user by cosine similarity.

    Args:
        store: The enrolled user_id -> embedding mapping.
        embedding: An L2-normalized query embedding.
        threshold: Minimum cosine similarity to accept a match.

    Returns:
        (user_id, score) for the best match, or ("unknown", best_score) if below threshold.
    """
    best_id = "unknown"
    best_score = -1.0
    for user_id, vec in store.items():
        score = float(np.dot(embedding, vec))
        if score > best_score:
            best_score = score
            best_id = user_id
    if best_score < threshold:
        return "unknown", best_score
    return best_id, best_score
