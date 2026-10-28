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

    def test_returning_to_auto_turns_on_for_counted_occupancy_with_camera_offline(self):
        self.automation.set_manual_state("OFF")
        self.automation.configure(mode="AUTO")

        status = self.automation.update(2, cameras_ready=False, now=10)

        self.assertEqual(self.dongle.commands, ["OFF", "ON"])
        self.assertEqual(status["light_state"], "ON")
        self.assertIsNone(status["off_timer_remaining_seconds"])
        self.assertIsNone(status["inhibit_reason"])

    def test_empty_room_timer_waits_for_all_cameras(self):
        self.automation.update(1, cameras_ready=True, now=0)
        status = self.automation.update(0, cameras_ready=False, now=10)

        self.assertEqual(self.dongle.commands, ["ON"])
        self.assertIsNone(status["off_timer_remaining_seconds"])
        self.assertEqual(status["inhibit_reason"], "Waiting for all cameras to be online")

    def test_manual_override_operates_without_camera_health(self):
        self.automation.configure(mode="FORCE_ON")
        self.automation.update(0, cameras_ready=False, now=0)
        self.assertEqual(self.dongle.commands, ["ON"])

    def test_manual_command_operates_without_enabling_automation(self):
        automation = LightAutomation(self.dongle, enabled=False)

        self.assertTrue(automation.set_manual_state("ON"))

        self.assertEqual(self.dongle.commands, ["ON"])
        status = automation.get_status()
        self.assertEqual(status["light_state"], "ON")
        self.assertEqual(status["mode"], "FORCE_ON")
        automation.update(0, cameras_ready=True, now=10)
        self.assertEqual(self.dongle.commands, ["ON"])

    def test_manual_off_override_is_not_reversed_by_occupancy(self):
        self.assertTrue(self.automation.set_manual_state("OFF"))

        self.automation.update(1, cameras_ready=True, now=10)

        self.assertEqual(self.dongle.commands, ["OFF"])
        self.assertEqual(self.automation.get_status()["mode"], "FORCE_OFF")

    def test_failed_manual_command_does_not_change_reported_state(self):
        class FailingDongle(FakeDongle):
            def set_state(inner_self, state):
                inner_self.commands.append(state)
                return False

        automation = LightAutomation(FailingDongle(), enabled=False)

        self.assertFalse(automation.set_manual_state("OFF"))
        self.assertEqual(automation.get_status()["light_state"], "UNKNOWN")

    def test_manual_command_rejects_unknown_state(self):
        with self.assertRaises(ValueError):
            self.automation.set_manual_state("toggle")

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