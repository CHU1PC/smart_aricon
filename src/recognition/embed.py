from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from loguru import logger

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "w600k_mbf.onnx"

_session = ort.InferenceSession(str(_MODEL_PATH), providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name
_output_name = _session.get_outputs()[0].name


def embed(aligned_bgr: cv2.typing.MatLike) -> npt.NDArray[np.float32]:
    """Computes an L2-normalized ArcFace embedding from an aligned 112x112 face.

    Args:
        aligned_bgr (cv2.typing.MatLike): A 112x112 aligned face crop in BGR (uint8).

    Returns:
        npt.NDArray[np.float32]: A 512-D L2-normalized embedding vector.
    """
    img = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 127.5
    blob = np.ascontiguousarray(np.transpose(img, (2, 0, 1))[None], dtype=np.float32)
    emb = _session.run([_output_name], {_input_name: blob})[0].reshape(-1)
    return emb / np.linalg.norm(emb)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from camera import capture_frames
    from recognition.detect import detect_and_align

    image = capture_frames(1)[0]
    face = detect_and_align(image)
    if face is None:
        logger.info("No face detected.")
    else:
        vec = embed(face)
        logger.info(f"embedding: shape={vec.shape}, norm={float(np.linalg.norm(vec)):.4f}")
