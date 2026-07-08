import cv2
from loguru import logger


def capture_frames(count: int) -> list[cv2.typing.MatLike]:
    """Captures a burst of frames from the default camera in one session.

    Args:
        count: Number of frames to capture.

    Returns:
        list[cv2.typing.MatLike]: The captured BGR frames (length == count).

    Raises:
        ValueError: If fewer than count frames could be captured.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        for _ in range(5):
            cap.read()
        frames: list[cv2.typing.MatLike] = []
        attempts = 0
        max_attempts = count * 5
        while len(frames) < count and attempts < max_attempts:
            attempts += 1
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        if len(frames) < count:
            err = f"Captured only {len(frames)}/{count} frames from the camera."
            logger.error(err)
            raise ValueError(err)
        return frames
    finally:
        cap.release()
