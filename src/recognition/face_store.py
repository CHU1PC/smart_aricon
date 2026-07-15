from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

_STORE_PATH = Path(__file__).resolve().parent.parent / "gallery.npz"
_EMBED_DIM = 512


def load(path: Path = _STORE_PATH) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Loads the face store as parallel (labels, embeddings) arrays.

    A user may appear on multiple rows (one template per row).

    Args:
        path: Path to the .npz store file.

    Returns:
        A (labels, embeddings) pair where labels[i] owns row embeddings[i].
    """
    if not path.exists():
        return [], np.empty((0, _EMBED_DIM), dtype=np.float32)
    with np.load(path) as data:
        raw_labels: npt.NDArray[np.str_] = data["labels"]
        embeddings: npt.NDArray[np.float32] = data["embeddings"]
    labels = [str(raw_labels[i]) for i in range(len(raw_labels))]
    return labels, embeddings


def add(
    labels: Sequence[str],
    embeddings: npt.NDArray[np.float32],
    user_id: str,
    embedding: npt.NDArray[np.float32],
) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Appends one template for a user.

    Args:
        labels: Existing per-row labels.
        embeddings: Existing (N, D) embedding matrix.
        user_id: The user to add a template for.
        embedding: The L2-normalized template to append.

    Returns:
        The updated (labels, embeddings).
    """
    new_labels = [*labels, user_id]
    new_embeddings = np.vstack([embeddings, embedding[None, :]]).astype(np.float32)
    return new_labels, new_embeddings


def save(
    labels: Sequence[str], embeddings: npt.NDArray[np.float32], path: Path = _STORE_PATH
) -> None:
    """Saves the (labels, embeddings) face store to disk.

    Args:
        labels: The per-row user labels.
        embeddings: The (N, D) embedding matrix.
        path: Path to the .npz store file.
    """
    np.savez(path, labels=np.array(labels), embeddings=embeddings)


def match(
    labels: Sequence[str],
    embeddings: npt.NDArray[np.float32],
    query: npt.NDArray[np.float32],
    threshold: float = 0.3,
) -> tuple[str, float]:
    """Finds the nearest enrolled template by cosine similarity.

    Args:
        labels: The per-row user labels.
        embeddings: The (N, D) L2-normalized embedding matrix.
        query: An L2-normalized query embedding.
        threshold: Minimum cosine similarity to accept a match.

    Returns:
        (user_id, score) of the nearest template, or ("unknown", best_score) if below threshold.
    """
    if len(labels) == 0:
        return "unknown", 0.0
    scores: npt.NDArray[np.float32] = embeddings @ query
    idx = int(np.argmax(scores))
    best = float(scores[idx])
    if best < threshold:
        return "unknown", best
    return labels[idx], best
