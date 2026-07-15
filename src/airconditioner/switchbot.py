import base64
import hashlib
import hmac
import json
import os
import time
import uuid

import httpx
from loguru import logger

_API_BASE = "https://api.switch-bot.com/v1.1"
_MODE_COOL = 2
_FAN_AUTO = 1
_STATUS_SUCCESS = 100


class SwitchBotClient:
    """Client for the SwitchBot Cloud API v1.1 air-conditioner control.

    In mock mode no request is sent; the command is only logged. This lets the
    whole flow run before a Hub/credentials are available.
    """

    def __init__(
        self, token: str, secret: str, device_id: str, *, mock: bool = True
    ) -> None:
        """Stores credentials and the mock flag.

        Args:
            token: SwitchBot API token.
            secret: SwitchBot API secret.
            device_id: SwitchBot device ID for the air conditioner.
            mock: If True, no request is sent; the command is only logged.

        Raises:
            ValueError: If mock is False and any of token, secret, or device_id is empty.
        """
        if not mock and not (token and secret and device_id):
            msg = "SwitchBot credentials are required when mock is disabled"
            raise ValueError(msg)
        self._token = token
        self._secret = secret
        self._device_id = device_id
        self._mock = mock

    @classmethod
    def from_env(cls) -> "SwitchBotClient":
        """Builds a client from SWITCHBOT_* environment variables.

        Reads SWITCHBOT_TOKEN / SWITCHBOT_SECRET / SWITCHBOT_DEVICE_ID and
        SWITCHBOT_MOCK (mock stays on unless it is "false"/"0"/"no").

        Returns:
            A configured SwitchBotClient.
        """
        mock = os.getenv("SWITCHBOT_MOCK", "true").lower() not in {"false", "0", "no"}
        return cls(
            os.getenv("SWITCHBOT_TOKEN", ""),
            os.getenv("SWITCHBOT_SECRET", ""),
            os.getenv("SWITCHBOT_DEVICE_ID", ""),
            mock=mock,
        )

    def _headers(self) -> dict[str, str]:
        t = str(int(time.time() * 1000))
        nonce = str(uuid.uuid4())
        payload = f"{self._token}{t}{nonce}".encode()
        digest = hmac.new(self._secret.encode(), payload, hashlib.sha256).digest()
        return {
            "Authorization": self._token,
            "sign": base64.b64encode(digest).decode(),
            "t": t,
            "nonce": nonce,
            "Content-Type": "application/json; charset=utf-8",
        }

    def set_air_conditioner(
        self, temperature: int, *, mode: int = _MODE_COOL, fan: int = _FAN_AUTO, power: str = "on"
    ) -> bool:
        """Sends a setAll command to the air conditioner.

        Args:
            temperature: Target temperature in Celsius.
            mode: 1 auto, 2 cool, 3 dry, 4 fan, 5 heat.
            fan: 1 auto, 2 low, 3 medium, 4 high.
            power: "on" or "off".

        Returns:
            True if accepted (statusCode 100), False otherwise. Mock mode always
            returns True.
        """
        parameter = f"{temperature},{mode},{fan},{power}"
        if self._mock:
            logger.info(f"[MOCK] SwitchBot setAll {parameter}")
            return True

        body = json.dumps(
            {"command": "setAll", "parameter": parameter, "commandType": "command"}
        ).encode()
        url = f"{_API_BASE}/devices/{self._device_id}/commands"
        try:
            response = httpx.post(url, headers=self._headers(), content=body, timeout=10)
            response.raise_for_status()
            result: dict[str, object] = response.json()
        except httpx.HTTPError as e:
            logger.error(f"SwitchBot request failed: {e}")
            return False

        if result.get("statusCode") != _STATUS_SUCCESS:
            logger.error(f"SwitchBot rejected command: {result}")
            return False
        logger.info(f"SwitchBot cooling set to {temperature}C")
        return True

    def get_devices(self) -> dict[str, object]:
        """Fetches the account's device list, ignoring mock mode.

        Returns:
            The parsed response body, or an empty dict on failure. The air
            conditioner appears under body.infraredRemoteList with its deviceId.
        """
        url = f"{_API_BASE}/devices"
        try:
            response = httpx.get(url, headers=self._headers(), timeout=10)
            response.raise_for_status()
            result: dict[str, object] = response.json()
        except httpx.HTTPError as e:
            logger.error(f"SwitchBot request failed: {e}")
            return {}
        return result
