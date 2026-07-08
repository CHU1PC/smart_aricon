import threading
from signal import pause

import cv2
from gpiozero import Button
from loguru import logger

from recognition.recognize import recognize

button = Button(25, pull_up=False, bounce_time=0.1)
_busy = threading.Lock()


def on_press() -> None:
    logger.info("Button pressed!")
    if not _busy.acquire(blocking=False):
        logger.info("Busy, ignoring press.")
        return
    try:
        user_id, score = recognize()
        if user_id == "unknown":
            logger.info(f"Unknown face (score={score:.3f}).")
        else:
            logger.info(f"Recognized '{user_id}' (score={score:.3f}).")
    except (ValueError, cv2.error) as e:
        logger.error(f"Recognition cycle failed: {e}")
    finally:
        _busy.release()


def main() -> None:
    button.when_pressed = on_press

    try:
        pause()
    except KeyboardInterrupt:
        logger.info("Exiting program...")
    finally:
        button.close()


if __name__ == "__main__":
    main()
