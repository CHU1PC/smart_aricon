from pathlib import Path

import cv2
import numpy as np
from loguru import logger

_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
)


_ARCFACE_SIZE = (112, 112)  # ArcFace input size

_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ],
    dtype=np.float32,
)

_detector = cv2.FaceDetectorYN.create(
    str(_MODEL_PATH),
    "",
    (320, 320),  # placeholder; updated per image via setInputSize()
    0.9,  # score_threshold
    0.3,  # nms_threshold
    5000,  # top_k
)


def detect_faces(image_bgr: cv2.typing.MatLike) -> cv2.typing.MatLike | None:
    """Detects faces in a BGR image using YuNet.

    Args:
        image_bgr (cv2.typing.MatLike): The input image in BGR format.

    Returns:
        cv2.typing.MatLike | None: An (N, 15) array of
            [x, y, w, h, 5x2 landmarks, score], or None if no face is found.
    """
    height, width = image_bgr.shape[:2]
    _detector.setInputSize((width, height))
    _, faces = _detector.detect(image_bgr)
    return faces


def largest_face(faces: cv2.typing.MatLike | None) -> cv2.typing.MatLike | None:
    """Returns the largest face by bounding-box area.

    Args:
        faces (cv2.typing.MatLike | None): The (N, 15) array from detect_faces.

    Returns:
        cv2.typing.MatLike | None: The row of the largest face, or None if empty.
    """
    if faces is None or len(faces) == 0:
        return None
    return faces[np.argmax(faces[:, 2] * faces[:, 3])]


def align_112(
    image_bgr: cv2.typing.MatLike, face_row: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    """Aligns and crops a face to 112x112 for ArcFace using its 5 landmarks.

    Args:
        image_bgr (cv2.typing.MatLike): The full BGR image.
        face_row (cv2.typing.MatLike): One row (15 values) from detect_faces.

    Returns:
        cv2.typing.MatLike: The aligned 112x112 BGR crop.
    """
    landmarks = face_row[4:14].reshape(5, 2).astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, _ARCFACE_DST, method=cv2.LMEDS)
    return cv2.warpAffine(image_bgr, matrix, _ARCFACE_SIZE, borderValue=0)


def detect_and_align(image_bgr: cv2.typing.MatLike) -> cv2.typing.MatLike | None:
    """Detects the largest face and returns its aligned 112x112 crop.

    Args:
        image_bgr (cv2.typing.MatLike): The input image in BGR format.

    Returns:
        cv2.typing.MatLike | None: The aligned 112x112 face, or None if no face is detected.
    """
    face = largest_face(detect_faces(image_bgr))
    if face is None:
        logger.info("No face detected.")
        return None
    return align_112(image_bgr, face)


if __name__ == "__main__":
    import sys

    # Allow `uv run recognition/detect.py`: put src/ on the path to import camera.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from camera import capture_image

    image = capture_image()
    aligned = detect_and_align(image)
    if aligned is None:
        logger.error("No face to align.")
    else:
        cv2.imwrite("aligned.jpg", aligned)
        logger.info("Saved aligned.jpg (112x112 aligned face)")
