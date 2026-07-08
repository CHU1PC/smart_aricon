from time import sleep

from gpiozero import LED
from loguru import logger

green = LED(22)
red = LED(23)


def main() -> None:
    try:
        while True:
            logger.info("GREEN on, RED off")
            green.on()
            red.off()
            sleep(1.5)
            logger.info("GREEN off, RED on")
            green.off()
            red.on()
            sleep(1.5)
    except KeyboardInterrupt:
        pass
    finally:
        green.off()
        red.off()
        green.close()
        red.close()


if __name__ == "__main__":
    main()
