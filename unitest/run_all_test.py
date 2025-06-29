"""
Runs all unittests in the 'unittest' directory.

All files matching the pattern 'test_*.py' will be discovered and executed.
"""

import unittest

from pathlib import Path

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=str(base_dir), pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)
