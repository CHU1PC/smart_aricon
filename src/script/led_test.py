from time import sleep

from gpiozero import LED
from loguru import logger

led = LED(22)


def _blink() -> None:
    while True:
        logger.info("LED on")
        led.on()
        sleep(1.5)
        logger.info("LED off")
        led.off()
        sleep(1.5)


def main() -> None:
    """Blinks the LED until interrupted, then cleans up."""
    try:
        _blink()
    except KeyboardInterrupt:
        pass
    finally:
        led.off()
        led.close()


if __name__ == "__main__":
    main()
