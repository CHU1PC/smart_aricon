from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from camera import capture_frames
from recognition import face_store
from recognition.detect import detect_and_align
from recognition.embed import embed

_FRAMES = 7


def recognize(threshold: float = 0.6, save_dir: Path | None = None) -> tuple[str, float]:
    """Recognizes a face from a short burst of camera frames.

    Args:
        threshold: Minimum cosine similarity to accept a match.
        save_dir: If given, saves each captured frame and its aligned face here.

    Returns:
        (user_id, score) for the nearest template, or ("unknown", best_score) if below
        threshold or if no face was found in any frame.
    """
    embeddings: list[npt.NDArray[np.float32]] = []
    for i, frame in enumerate(capture_frames(_FRAMES)):
        face = detect_and_align(frame)
        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"frame_{i:02d}.jpg"), frame)
            if face is not None:
                cv2.imwrite(str(save_dir / f"face_{i:02d}.jpg"), face)
        if face is not None:
            embeddings.append(embed(face))

    if not embeddings:
        logger.info("No face detected in any frame.")
        return "unknown", 0.0

    mean = np.mean(embeddings, axis=0)
    query: npt.NDArray[np.float32] = (mean / np.linalg.norm(mean)).astype(np.float32)
    labels, gallery = face_store.load()
    return face_store.match(labels, gallery, query, threshold)


if __name__ == "__main__":
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
    user_id, score = recognize(save_dir=debug_dir)
    logger.info(f"Recognized: {user_id} (score={score:.3f})")
