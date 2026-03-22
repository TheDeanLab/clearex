from __future__ import annotations

import sys

from setuptools import setup

if sys.version_info[:2] != (3, 12):
    detected = ".".join(str(part) for part in sys.version_info[:3])
    raise RuntimeError(
        "ClearEx currently supports Python 3.12.x only. "
        f"Detected Python {detected}. "
        "BaSiCPy currently depends on scipy<1.13, and SciPy 1.12 does not "
        "provide a supported Python 3.13 wheel path for this project."
    )


setup()
