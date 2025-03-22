# Set test environment
import argparse
import os
import sys

# Add the parent directory to path to help Python find the modules
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

os.environ["ENV"] = "e2etest"

from tests.e2e_tests.e2e_connection import (
    failed_connection,
    successful_connection,
)
from tests.e2e_tests.e2e_excel import run_plate_with_correct_excel

# Dictionary mapping test names to functions
TEST_FUNCTIONS = {
    "failed_connection": failed_connection,
    "successful_connection": successful_connection,
    "e2e_excel": run_plate_with_correct_excel,
}


def main() -> int:
    """Main entry point for the integration tests"""
    parser = argparse.ArgumentParser(
        description="Run metadata integration tests"
    )
    parser.add_argument(
        "test", choices=TEST_FUNCTIONS.keys(), help="Test to run"
    )
    args = parser.parse_args()

    try:
        print(f"\nRunning test: {args.test}")
        print(f"Description: {TEST_FUNCTIONS[args.test].__doc__}\n")
        TEST_FUNCTIONS[args.test]()
        return 0
    except Exception as e:  # noqa: BLE001
        # The rich error will have already been displayed
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
