import argparse

import numpy as np
import numpy.typing as npt
from loguru import logger

from camera import capture_frames
from recognition import face_store
from recognition.detect import detect_and_align
from recognition.embed import embed


def register(user_id: str, count: int) -> None:
    """Captures a burst and appends one face template for the user.

    Args:
        user_id: The identifier to enroll under.
        count: Number of frames to capture in one burst.
    """
    embeddings: list[npt.NDArray[np.float32]] = []
    for frame in capture_frames(count):
        face = detect_and_align(frame)
        if face is not None:
            embeddings.append(embed(face))

    if not embeddings:
        logger.error("No face detected in any frame; aborting")
        return

    mean = np.mean(embeddings, axis=0)
    template: npt.NDArray[np.float32] = (mean / np.linalg.norm(mean)).astype(np.float32)
    labels, gallery = face_store.load()
    labels, gallery = face_store.add(labels, gallery, user_id, template)
    face_store.save(labels, gallery)
    logger.info(f"Registered '{user_id}' ({labels.count(user_id)} templates, {len(labels)} total)")


def main() -> None:
    """Command-line interface for registering a user's face."""
    parser = argparse.ArgumentParser(description="Register a user's face into the store")
    parser.add_argument("--user", required=True)
    parser.add_argument("--count", type=int, default=15)
    args = parser.parse_args()
    register(args.user, args.count)


if __name__ == "__main__":
    main()
