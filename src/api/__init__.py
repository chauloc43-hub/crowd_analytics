"""Small, stateful FastAPI surface for the Crowd Analytics demo.

The package deliberately contains no import-time model construction.  Use
``create_api_app`` to obtain an ASGI application, or inject a lightweight
session manager in tests and alternate deployments.
"""

from src.api.app import create_api_app
from src.api.config import ApiSettings
from src.api.sessions import DemoSessionManager

__all__ = ["ApiSettings", "DemoSessionManager", "create_api_app"]
