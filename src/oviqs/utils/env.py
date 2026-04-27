from __future__ import annotations

import platform
import sys


def environment_block() -> dict[str, str]:
    return {"python_version": sys.version, "platform": platform.platform()}
