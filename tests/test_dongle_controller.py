import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dongle_controller import DongleController


class DongleControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            requests = []
            response_status = 200
            status_payload = {"lastStatus": "on", "lastSeenAt": "2026-10-27T12:00:00Z"}

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                self.requests.append((self.path, self.headers, body))
                self.send_response(self.response_status)
                self.end_headers()

            def do_GET(self):
                self.requests.append((self.path, self.headers, None))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(self.status_payload).encode("utf-8"))

            def log_message(self, format, *args):
                pass

        cls.handler = Handler
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()

    def setUp(self):
        self.handler.requests.clear()
        self.handler.response_status = 200
        self.handler.status_payload = {
            "lastStatus": "on",
            "lastSeenAt": "2026-10-27T12:00:00Z",
        }
        host, port = self.server.server_address
        self.config = {
            "api_url": f"http://{host}:{port}/devices",
            "device_id": "A1026002",
            "thing_id": "81f9a6fb-55ad-43e3-bbea-fb7e8c7a2dfb",
            "api_key": "test-key",
            "auth_token": "test-token",
        }
        self.controller = DongleController(
            retry_count=1,
            retry_delay=0,
            api_config=self.config,
        )

    def test_api_control_sends_configured_switch_payload(self):
        self.config["device_id"] = "IOTIQDC2_A1026002"
        self.assertTrue(self.controller.set_state("OFF"))

        path, headers, body = self.handler.requests[0]
        self.assertEqual(path, "/iot/devices/A1026002/control")
        self.assertEqual(headers["Authorization"], "Bearer test-token")
        self.assertEqual(headers["X-API-Key"], "test-key")
        self.assertEqual(body["parameters"]["device_id"], "A1026002")
        self.assertEqual(body["parameters"]["thing_id"], self.config["thing_id"])
        self.assertEqual(body["parameters"]["status"], "off")
        self.assertEqual(body["payload"]["deviceid"], "IOTIQDC2_A1026002")
        self.assertEqual(body["payload"]["switch_no"], "S1")
        self.assertEqual(body["subTopic"], "control")
        self.assertEqual(
            body["topic"],
            f"mqtt/device/{self.config['thing_id']}/control",
        )
        self.assertEqual(self.controller.get_status()["status"], "READY")

    def test_hive_catalog_control_sends_policy_command(self):
        self.config.update({
            "api_url": f"http://{self.server.server_address[0]}:{self.server.server_address[1]}/api",
            "device_id": "A1026002",
            "device_name": "IOTIQDC2_A1026002",
            "catalog_device_id": "catalog-device",
            "command_key": "dongle-control-policy",
            "message_id": "control-message",
        })
        self.config.pop("api_key")
        self.config.pop("auth_token")
        controller = DongleController(retry_count=1, retry_delay=0, api_config=self.config)

        self.assertTrue(controller.set_state("OFF"))

        path, headers, body = self.handler.requests[0]
        self.assertEqual(
            path,
            "/api/iot/catalog/devices/catalog-device/commands/dongle-control-policy",
        )
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body["messageId"], "control-message")
        self.assertEqual(body["parameters"]["device_id"], "A1026002")
        self.assertEqual(body["parameters"]["channel"], "1")
        self.assertEqual(body["parameters"]["switch_no"], "S1")
        self.assertEqual(body["parameters"]["status"], "off")
        self.assertEqual(body["payload"], {
            "deviceid": "IOTIQDC2_A1026002",
            "channel": "1",
            "switch_no": "S1",
            "status": "off",
        })
        self.assertEqual(body["topic"], f"mqtt/device/{self.config['thing_id']}/control")
        self.assertEqual(body["subTopic"], "control")
        self.assertEqual(controller.get_status()["status"], "READY")

    def test_reads_reported_light_state_from_hive(self):
        self.handler.status_payload = {
            "lastStatus": "off",
            "lastSeenAt": "2026-10-27T12:05:00Z",
        }

        status = self.controller.get_status()

        self.assertEqual(status["reported_state"], "OFF")
        self.assertEqual(status["reported_at"], "2026-10-27T12:05:00Z")
        self.assertEqual(status["reported_at_source"], "device_report")
        self.assertIsNone(status["reported_error"])
        self.assertEqual(self.handler.requests[0][0], "/devices/iot/devices/A1026002")

    def test_marks_record_update_as_not_live_telemetry(self):
        self.handler.status_payload = {
            "lastStatus": "on",
            "lastSeenAt": None,
            "updatedAt": "2026-10-27T12:10:00Z",
        }

        status = self.controller.get_status()

        self.assertEqual(status["reported_state"], "ON")
        self.assertEqual(status["reported_at_source"], "record_updated")

    def test_unknown_hive_state_is_reported_as_unknown(self):
        self.handler.status_payload = {"lastStatus": None}

        status = self.controller.get_status()

        self.assertEqual(status["reported_state"], "UNKNOWN")
        self.assertEqual(status["reported_error"], "Hive has not reported a light state")

    def test_api_error_is_reported(self):
        self.handler.response_status = 401

        self.assertFalse(self.controller.set_state("ON"))
        status = self.controller.get_status()
        self.assertEqual(status["status"], "ERROR")
        self.assertEqual(status["error"], "HTTP control request failed (HTTP 401)")

    def test_api_requires_device_and_thing_ids(self):
        controller = DongleController(api_config={"api_url": self.config["api_url"]})

        self.assertFalse(controller.set_state("ON"))
        status = controller.get_status()
        self.assertEqual(status["status"], "ERROR")
        self.assertEqual(status["error"], "ON command is not configured")

    def test_api_requires_authentication_before_sending(self):
        config = {key: value for key, value in self.config.items() if key not in {"api_key", "auth_token"}}
        controller = DongleController(api_config=config)

        self.assertFalse(controller.set_state("ON"))
        self.assertEqual(self.handler.requests, [])
        self.assertEqual(controller.get_status()["error"], "API authentication is not configured")

    def test_rejects_unknown_light_state(self):
        with self.assertRaises(ValueError):
            self.controller.set_state("toggle")


if __name__ == "__main__":
    unittest.main()
