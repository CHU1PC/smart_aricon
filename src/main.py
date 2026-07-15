import threading
from signal import pause

import cv2
from gpiozero import LED, Button
from loguru import logger

from airconditioner.switchbot import SwitchBotClient
from recognition.recognize import recognize
from temperature import DHT11, DHT11Error

button = Button(25, pull_up=False, bounce_time=0.1)
led = LED(22)
dht = DHT11(14)
sb = SwitchBotClient.from_env()
_busy = threading.Lock()

SETPOINTS = {"Tadashi": 25}


def _handle_press() -> None:
    led.off()
    user_id, score = recognize()
    if user_id == "unknown":
        logger.info(f"Unknown face (score={score:.3f}).")
        return

    led.blink(on_time=3, off_time=0, n=1, background=True)
    logger.info(f"Recognized '{user_id}' (score={score:.3f}).")

    try:
        humidity, temperature = dht.read_data()
    except DHT11Error as e:
        logger.error(f"Temperature read failed: {e}")
        return

    setpoint = SETPOINTS.get(user_id, 25)
    logger.info(f"Room {temperature}C (humidity {humidity}%), setpoint {setpoint}C.")
    if temperature > setpoint:
        logger.info("Cooling needed.")
        sb.set_air_conditioner(setpoint)
    else:
        logger.info("Within range; no action.")


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
