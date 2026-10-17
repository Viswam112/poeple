import os
import shlex
import subprocess
import threading
import time


class DongleController:
    """Run configured local bridge commands; hardware-specific protocol stays in the bridge."""

    def __init__(self, on_command="", off_command="", retry_count=3, retry_delay=1.0, timeout=10.0):
        self._commands = {
            "ON": self._parse_command(on_command),
            "OFF": self._parse_command(off_command),
        }
        self.retry_count = max(1, int(retry_count))
        self.retry_delay = max(0.0, float(retry_delay))
        self.timeout = max(0.1, float(timeout))
        self._lock = threading.Lock()
        self._last_error = None
        self._last_command_at = None
        self._last_command = None

    @staticmethod
    def _parse_command(command):
        if not command:
            return []
        return shlex.split(command, posix=os.name != "nt")

    def set_state(self, state):
        command = self._commands.get(state)
        with self._lock:
            if not command:
                self._last_error = f"{state} command is not configured"
                self._last_command = state
                self._last_command_at = time.time()
                return False

            for attempt in range(self.retry_count):
                try:
                    subprocess.run(command, check=True, timeout=self.timeout, capture_output=True, text=True)
                    self._last_error = None
                    self._last_command = state
                    self._last_command_at = time.time()
                    return True
                except (OSError, subprocess.SubprocessError) as exc:
                    self._last_error = str(exc)
                    if attempt + 1 < self.retry_count:
                        time.sleep(self.retry_delay)

            self._last_command = state
            self._last_command_at = time.time()
            return False

    def get_status(self):
        with self._lock:
            configured = all(self._commands.values())
            if self._last_error:
                status = "ERROR"
            else:
                status = "READY" if configured else "NOT_CONFIGURED"
            return {
                "status": status,
                "last_command": self._last_command,
                "last_command_at": self._last_command_at,
                "error": self._last_error,
            }