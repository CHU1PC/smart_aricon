import json

from loguru import logger

from airconditioner.switchbot import SwitchBotClient


def main() -> None:
    """Prints the SwitchBot device list so the air-conditioner deviceId can be found.

    The air conditioner is an infrared device: look under body.infraredRemoteList.
    """
    devices = SwitchBotClient.from_env().get_devices()
    if not devices:
        logger.error("No response; check SWITCHBOT_TOKEN / SWITCHBOT_SECRET.")
        return
    logger.info(json.dumps(devices, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
