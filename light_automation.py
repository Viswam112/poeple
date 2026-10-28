import threading
import time


class LightAutomation:
    MODES = {"AUTO", "FORCE_ON", "FORCE_OFF"}
    ALLOWED_DELAYS = {60, 120, 300, 600, 900, 1800}
    COMMAND_RETRY_COOLDOWN_SECONDS = 30

    def __init__(self, dongle, enabled=False, off_delay_seconds=300):
        self.dongle = dongle
        self._lock = threading.RLock()
        self._enabled = bool(enabled)
        self._mode = "AUTO"
        self._off_delay_seconds = self._validate_delay(off_delay_seconds)
        self._light_state = "UNKNOWN"
        self._zero_since = None
        self._has_seen_presence = False
        self._last_attempted_state = None
        self._last_attempt_at = None
        self._inhibit_reason = "Waiting for cameras"

    @classmethod
    def _validate_delay(cls, seconds):
        seconds = int(seconds)
        if seconds not in cls.ALLOWED_DELAYS:
            raise ValueError("off_delay_seconds must be one of: 60, 120, 300, 600, 900, 1800")
        return seconds

    def configure(self, enabled=None, mode=None, off_delay_seconds=None):
        with self._lock:
            if enabled is not None:
                enabled = bool(enabled)
                if enabled != self._enabled:
                    self._enabled = enabled
                    self._last_attempted_state = None
                    self._last_attempt_at = None
            if mode is not None:
                mode = str(mode).upper()
                if mode not in self.MODES:
                    raise ValueError("mode must be AUTO, FORCE_ON, or FORCE_OFF")
                if mode != self._mode:
                    self._mode = mode
                    self._last_attempted_state = None
                    self._last_attempt_at = None
            if off_delay_seconds is not None:
                self._off_delay_seconds = self._validate_delay(off_delay_seconds)
            return self._snapshot(time.monotonic())

    def set_manual_state(self, state):
        state = str(state).upper()
        if state not in {"ON", "OFF"}:
            raise ValueError("state must be ON or OFF")
        with self._lock:
            if not self.dongle.set_state(state):
                return False
            self._light_state = state
            self._mode = "FORCE_ON" if state == "ON" else "FORCE_OFF"
            self._last_attempted_state = None
            self._last_attempt_at = None
            self._zero_since = None
            self._inhibit_reason = None
            return True

    def update(self, occupancy, cameras_ready, now=None):
        now = time.monotonic() if now is None else now
        occupancy = max(0, int(occupancy))
        with self._lock:
            if occupancy > 0:
                self._has_seen_presence = True
                self._zero_since = None

            if self._mode == "FORCE_ON":
                self._inhibit_reason = None
                self._request_state("ON", now)
                return self._snapshot(now)
            if self._mode == "FORCE_OFF":
                self._inhibit_reason = None
                self._request_state("OFF", now)
                return self._snapshot(now)

            if not self._enabled:
                self._zero_since = None
                self._inhibit_reason = "Automatic lighting is disabled"
                return self._snapshot(now)
            if occupancy > 0:
                self._inhibit_reason = None
                self._request_state("ON", now)
                return self._snapshot(now)
            if not cameras_ready:
                self._zero_since = None
                self._inhibit_reason = "Waiting for all cameras to be online"
                return self._snapshot(now)

            self._inhibit_reason = None
            if self._has_seen_presence:
                if self._zero_since is None:
                    self._zero_since = now
                if now - self._zero_since >= self._off_delay_seconds:
                    self._request_state("OFF", now)
            return self._snapshot(now)

    def _request_state(self, state, now):
        if self._light_state == state:
            return
        if (self._last_attempted_state == state and self._last_attempt_at is not None
                and now - self._last_attempt_at < self.COMMAND_RETRY_COOLDOWN_SECONDS):
            return
        self._last_attempted_state = state
        self._last_attempt_at = now
        if self.dongle.set_state(state):
            self._light_state = state

    def _snapshot(self, now):
        remaining = None
        if self._mode == "AUTO" and self._enabled and self._zero_since is not None:
            remaining = max(0, int(self._off_delay_seconds - (now - self._zero_since)))
        return {
            "enabled": self._enabled,
            "mode": self._mode,
            "off_delay_seconds": self._off_delay_seconds,
            "light_state": self._light_state,
            "off_timer_remaining_seconds": remaining,
            "inhibit_reason": self._inhibit_reason,
            "dongle": self.dongle.get_status(),
        }

    def get_status(self):
        with self._lock:
            return self._snapshot(time.monotonic())