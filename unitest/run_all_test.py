"""Utility to run all tests in the :mod:`unittest` package.

This script uses ``pytest`` so that both ``unittest`` style classes and
standalone test functions are discovered correctly. It also ensures the
project root is on ``sys.path`` so that local modules such as ``config`` can be
imported without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    # Add the repository root to ``sys.path`` for local imports.
    sys.path.insert(0, str(base_dir.parent))
    # Run tests located in the current base_dir
    sys.exit(pytest.main([str(base_dir)]))
