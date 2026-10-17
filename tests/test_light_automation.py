import unittest

from light_automation import LightAutomation


class FakeDongle:
    def __init__(self):
        self.commands = []

    def set_state(self, state):
        self.commands.append(state)
        return True

    def get_status(self):
        return {"status": "READY", "error": None}


class LightAutomationTests(unittest.TestCase):
    def setUp(self):
        self.dongle = FakeDongle()
        self.automation = LightAutomation(self.dongle, enabled=True, off_delay_seconds=60)

    def test_zero_at_start_does_not_turn_lights_off(self):
        self.automation.update(0, cameras_ready=True, now=0)
        self.automation.update(0, cameras_ready=True, now=120)
        self.assertEqual(self.dongle.commands, [])

    def test_timeout_turns_off_once(self):
        self.automation.update(1, cameras_ready=True, now=0)
        self.automation.update(0, cameras_ready=True, now=10)
        self.automation.update(0, cameras_ready=True, now=69)
        self.assertEqual(self.dongle.commands, ["ON"])
        self.automation.update(0, cameras_ready=True, now=70)
        self.automation.update(0, cameras_ready=True, now=90)
        self.assertEqual(self.dongle.commands, ["ON", "OFF"])

    def test_returning_person_cancels_off_timer(self):
        self.automation.update(1, cameras_ready=True, now=0)
        self.automation.update(0, cameras_ready=True, now=10)
        self.automation.update(1, cameras_ready=True, now=50)
        self.automation.update(0, cameras_ready=True, now=60)
        self.automation.update(0, cameras_ready=True, now=119)
        self.assertEqual(self.dongle.commands, ["ON"])
        self.automation.update(0, cameras_ready=True, now=120)
        self.assertEqual(self.dongle.commands, ["ON", "OFF"])

    def test_camera_outage_prevents_auto_off_and_restarts_delay(self):
        self.automation.update(1, cameras_ready=True, now=0)
        self.automation.update(0, cameras_ready=True, now=10)
        self.automation.update(0, cameras_ready=False, now=50)
        self.automation.update(0, cameras_ready=True, now=100)
        self.assertEqual(self.dongle.commands, ["ON"])
        self.automation.update(0, cameras_ready=True, now=159)
        self.assertEqual(self.dongle.commands, ["ON"])
        self.automation.update(0, cameras_ready=True, now=160)
        self.assertEqual(self.dongle.commands, ["ON", "OFF"])

    def test_manual_override_operates_without_camera_health(self):
        self.automation.configure(mode="FORCE_ON")
        self.automation.update(0, cameras_ready=False, now=0)
        self.assertEqual(self.dongle.commands, ["ON"])

    def test_failed_command_retries_after_cooldown(self):
        class RecoveringDongle(FakeDongle):
            def set_state(inner_self, state):
                inner_self.commands.append(state)
                return len(inner_self.commands) > 1

        dongle = RecoveringDongle()
        automation = LightAutomation(dongle, enabled=True, off_delay_seconds=60)
        automation.update(1, cameras_ready=True, now=0)
        automation.update(1, cameras_ready=True, now=10)
        self.assertEqual(dongle.commands, ["ON"])
        automation.update(1, cameras_ready=True, now=30)
        self.assertEqual(dongle.commands, ["ON", "ON"])


if __name__ == "__main__":
    unittest.main()