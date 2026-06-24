from signal import pause

from gpiozero import Button
from loguru import logger

button = Button(17, pull_up=True, bounce_time=0.1)


def on_press() -> None:
    logger.info("Button pressed!")


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
