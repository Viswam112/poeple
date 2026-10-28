# poeple

## Lighting dongle configuration

The lighting controller can use Hive Connect's catalog command API, another
authenticated provider HTTP API, or local ON/OFF bridge commands. For Hive
Connect, set the API base URL, device and catalog IDs, command key, message ID,
Thing ID, and switch mapping in `config/dongle_api.json`, using
`config/dongle_api.example.json` as a template. The local config is excluded
from Git so API keys and tokens are not committed.

Hive Connect control sends a catalog command to
`POST /iot/catalog/devices/{catalog_device_id}/commands/{command_key}` using
the message ID and MQTT payload/topic fields from the device's communication
policy. Other provider HTTP control uses
`POST /iot/devices/{device_id}/control`. API keys use
the configured `api_key_header` (default `X-API-Key`); tokens use the configured
`auth_scheme` (default `Bearer`). Keep credentials local and never commit them.
If the provider requires a different endpoint or authentication header, update
the corresponding local configuration values.

The dashboard separately shows the last state stored by Hive and the last
command sent by this application. A Hive `lastSeenAt` timestamp is a device
report; when only `updatedAt` is available, the UI labels it as a record update
and warns that it is not live confirmation.

Automatic lighting is enabled by default; set `LIGHT_AUTO_ENABLED=false` to
disable it at startup. The dashboard also provides manual **Turn on** and
**Turn off** buttons. Manual commands set a FORCE_ON or FORCE_OFF override so
automatic occupancy updates do not immediately reverse the button command;
select Auto to return to occupancy-based control. In Auto mode, any positive
count from an online camera requests ON, even if another camera is offline.
Automatic OFF is more conservative: its delay only runs when every enabled
camera is online, so an offline camera cannot make an occupied room appear
empty. The controller will not send unauthenticated requests through the
generic provider API. Configure any provider credentials only in the ignored
local `config/dongle_api.json`; the current controller does not support
certificate-based API authentication.