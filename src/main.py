from signal import pause

import cv2
from gpiozero import Button
from loguru import logger

from camera import capture_frames

button = Button(25, pull_up=False, bounce_time=0.1)


def on_press() -> None:
    logger.info("Button pressed!")

    try:
        image = capture_frames(1)[0]
        logger.info("Image captured successfully.")
        cv2.imwrite("captured_image.jpg", image)  # Sample Code
    except (ValueError, cv2.error) as e:
        logger.error(f"Error capturing image: {e}")


def main() -> None:
    button.when_pressed = on_press  # When the button is pressed, call the on_press function

    try:
        pause()
    except KeyboardInterrupt:
        logger.info("Exiting program...")
    finally:
        button.close()


if __name__ == "__main__":
    main()
