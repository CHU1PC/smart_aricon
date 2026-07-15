import threading
from signal import pause

import cv2
from gpiozero import LED, Button
from loguru import logger

from recognition.recognize import recognize

button = Button(25, pull_up=False, bounce_time=0.1)
led = LED(22)
_busy = threading.Lock()


def _handle_press() -> None:
    led.off()
    user_id, score = recognize()
    if user_id == "unknown":
        logger.info(f"Unknown face (score={score:.3f}).")
    else:
        led.on()
        logger.info(f"Recognized '{user_id}' (score={score:.3f}).")


def on_press() -> None:
    logger.info("Button pressed!")
    if not _busy.acquire(blocking=False):
        logger.info("Busy, ignoring press.")
        return
    try:
        _handle_press()
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
        led.close()


if __name__ == "__main__":
    main()
