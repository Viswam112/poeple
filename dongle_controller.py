import json
import os
import shlex
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


class DongleController:
    """Control lights through a local bridge command or the provider's HTTP API."""

    def __init__(self, on_command="", off_command="", retry_count=3, retry_delay=1.0, timeout=10.0,
                 api_config=None):
        self._commands = {
            "ON": self._parse_command(on_command),
            "OFF": self._parse_command(off_command),
        }
        self._api_config = api_config or {}
        self.retry_count = max(1, int(retry_count))
        self.retry_delay = max(0.0, float(retry_delay))
        self.timeout = max(0.1, float(timeout))
        self._lock = threading.Lock()
        self._last_error = None
        self._last_command_at = None
        self._last_command = None
        self._status_lock = threading.Lock()
        self._status_poll_interval = 5.0
        self._status_checked_at = 0.0
        self._reported_state = "UNKNOWN"
        self._reported_at = None
        self._reported_at_source = None
        self._reported_error = None

    @staticmethod
    def _parse_command(command):
        if not command:
            return []
        return shlex.split(command, posix=os.name != "nt")

    def _api_ready(self):
        return (
            self._hive_api_ready()
            or (
                all(self._api_config.get(key) for key in ("api_url", "device_id", "thing_id"))
                and bool(self._api_config.get("auth_token") or self._api_config.get("api_key"))
            )
        )

    def _hive_api_ready(self):
        return all(
            self._api_config.get(key)
            for key in ("api_url", "device_id", "thing_id", "catalog_device_id", "command_key", "message_id")
        )

    def _send_hive_command(self, state):
        config = self._api_config
        endpoint = config.get(
            "control_path",
            "/iot/catalog/devices/{catalog_device_id}/commands/{command_key}",
        ).format(
            catalog_device_id=quote(str(config["catalog_device_id"]), safe=""),
            command_key=quote(str(config["command_key"]), safe=""),
        )
        url = urljoin(config["api_url"].rstrip("/") + "/", endpoint.lstrip("/"))
        device_id = str(config["device_id"])
        device_name = str(config.get("device_name") or f"IOTIQDC2_{device_id.removeprefix('IOTIQDC2_')}")
        thing_id = str(config["thing_id"])
        channel = str(config.get("channel", "1"))
        switch_no = str(config.get("switch_no", "S1"))
        sub_topic = str(config.get("sub_topic", "control"))
        status = state.lower()

        body = {
            "messageId": str(config["message_id"]),
            "parameters": {
                "deviceid": device_id,
                "device_id": device_id,
                "thingid": thing_id,
                "thing_id": thing_id,
                "thingname": thing_id,
                "channel": channel,
                "switch_no": switch_no,
                "status": status,
            },
            "payload": {
                "deviceid": device_name,
                "channel": channel,
                "switch_no": switch_no,
                "status": status,
            },
            "topic": f"mqtt/device/{thing_id}/{sub_topic}",
            "subTopic": sub_topic,
        }
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            if not 200 <= response.status < 300:
                raise OSError(f"Hive Connect control request failed (HTTP {response.status})")

    def _send_api_command(self, state):
        config = self._api_config
        endpoint = config.get("control_path", "/iot/devices/{device_id}/control")
        configured_device_id = str(config["device_id"])
        device_id = configured_device_id.removeprefix("IOTIQDC2_")
        endpoint = endpoint.format(device_id=quote(device_id, safe=""))
        url = urljoin(config["api_url"].rstrip("/") + "/", endpoint)
        thing_id = str(config["thing_id"])
        channel = str(config.get("channel", "1"))
        switch_no = str(config.get("switch_no", "S1"))
        sub_topic = str(config.get("sub_topic", "control"))
        status = state.lower()

        body = {
            "parameters": {
                "channel": channel,
                "switch_no": switch_no,
                "status": status,
                "deviceid": device_id,
                "device_id": device_id,
                "thing_id": thing_id,
                "thingid": thing_id,
                "thingname": thing_id,
            },
            "payload": {
                "deviceid": f"IOTIQDC2_{device_id}",
                "channel": channel,
                "switch_no": switch_no,
                "status": status,
            },
            "subTopic": sub_topic,
            "topic": f"mqtt/device/{thing_id}/{sub_topic}",
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        auth_token = config.get("auth_token")
        api_key = config.get("api_key")
        if auth_token:
            scheme = config.get("auth_scheme", "Bearer").strip()
            headers["Authorization"] = f"{scheme} {auth_token}" if scheme else auth_token
        if api_key:
            headers[config.get("api_key_header", "X-API-Key")] = api_key

        request = Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            if not 200 <= response.status < 300:
                raise OSError(f"HTTP control request failed (HTTP {response.status})")

    def _refresh_reported_state(self):
        config = self._api_config
        if not all(config.get(key) for key in ("api_url", "device_id")):
            return

        with self._status_lock:
            now = time.monotonic()
            if now - self._status_checked_at < self._status_poll_interval:
                return
            self._status_checked_at = now

            device_id = str(config["device_id"]).removeprefix("IOTIQDC2_")
            endpoint = config.get("status_path", "/iot/devices/{device_id}").format(
                device_id=quote(device_id, safe="")
            )
            url = urljoin(config["api_url"].rstrip("/") + "/", endpoint.lstrip("/"))
            request = Request(url, headers={"Accept": "application/json"}, method="GET")
            try:
                with urlopen(request, timeout=min(self.timeout, 3.0)) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Hive status response must be a JSON object")
                state = str(data.get("lastStatus") or "").strip().upper()
                if state in {"ON", "1", "TRUE", "ACTIVE"}:
                    self._reported_state = "ON"
                elif state in {"OFF", "0", "FALSE", "INACTIVE"}:
                    self._reported_state = "OFF"
                else:
                    self._reported_state = "UNKNOWN"
                self._reported_at = data.get("lastSeenAt") or data.get("updatedAt")
                self._reported_at_source = (
                    "device_report" if data.get("lastSeenAt") else
                    "record_updated" if data.get("updatedAt") else None
                )
                self._reported_error = None if state else "Hive has not reported a light state"
            except HTTPError as exc:
                self._reported_error = f"Hive status request failed (HTTP {exc.code})"
            except (OSError, URLError, ValueError, json.JSONDecodeError):
                self._reported_error = "Could not read the latest Hive light status"

    def set_state(self, state):
        state = str(state).upper()
        if state not in {"ON", "OFF"}:
            raise ValueError("state must be ON or OFF")

        command = self._commands[state]
        with self._lock:
            if command:
                send = lambda: subprocess.run(
                    command, check=True, timeout=self.timeout, capture_output=True, text=True
                )
            elif self._hive_api_ready():
                send = lambda: self._send_hive_command(state)
            elif self._api_ready():
                send = lambda: self._send_api_command(state)
            else:
                has_api_target = all(
                    self._api_config.get(key) for key in ("api_url", "device_id", "thing_id")
                )
                if has_api_target and any(
                    self._api_config.get(key)
                    for key in ("catalog_device_id", "command_key", "message_id")
                ):
                    self._last_error = "Hive Connect control configuration is incomplete"
                elif has_api_target:
                    self._last_error = "API authentication is not configured"
                else:
                    self._last_error = f"{state} command is not configured"
                self._last_command = state
                self._last_command_at = time.time()
                return False

            for attempt in range(self.retry_count):
                try:
                    send()
                    self._last_error = None
                    self._last_command = state
                    self._last_command_at = time.time()
                    return True
                except HTTPError as exc:
                    self._last_error = f"HTTP control request failed (HTTP {exc.code})"
                except (OSError, subprocess.SubprocessError, URLError):
                    self._last_error = "Control request failed"
                if attempt + 1 < self.retry_count:
                    time.sleep(self.retry_delay)

            self._last_command = state
            self._last_command_at = time.time()
            return False

    def get_status(self):
        self._refresh_reported_state()
        with self._lock:
            configured = all(self._commands.values()) or self._api_ready()
            if self._last_error:
                status = "ERROR"
            else:
                status = "READY" if configured else "NOT_CONFIGURED"
            result = {
                "status": status,
                "last_command": self._last_command,
                "last_command_at": self._last_command_at,
                "error": self._last_error,
            }
        with self._status_lock:
            result.update({
                "reported_state": self._reported_state,
                "reported_at": self._reported_at,
                "reported_at_source": self._reported_at_source,
                "reported_error": self._reported_error,
            })
        return result
