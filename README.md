# poeple

## Dongle API configuration

Keep local dongle API details in `config/dongle_api.json`, using
`config/dongle_api.example.json` as a template. The local config is excluded
from Git so API keys and tokens are not committed.

This file is a place to store the provider details; the current lighting
controller still uses local ON/OFF bridge commands. API control is not active
until the dongle's API protocol is integrated.