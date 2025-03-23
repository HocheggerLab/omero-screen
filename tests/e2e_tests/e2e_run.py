"""
This script is used to run the integration tests.
It will create a plate, run the test, and then tear down the plate.
The test can be chosen from the command line.
The teardown can be chosen with the --teardown flag with yes or no as options.
The default is yes.
Run the script with omero-integration-test test_name
New tests should be added to the TEST_FUNCTIONS dictionary.
"""

# Set test environment
import argparse
import os
import sys

os.environ["ENV"] = "e2etest"
from omero.gateway import BlitzGateway
from omero_utils.omero_connect import omero_connect
from omero_utils.omero_plate import cleanup_plate

from tests.e2e_tests.e2e_connection import (
    failed_connection,
    successful_connection,
)
from tests.e2e_tests.e2e_excel import run_plate_with_correct_excel
from tests.e2e_tests.e2e_setup import e2e_excel_setup

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
    parser.add_argument(
        "-t",
        "--teardown",
        choices=["yes", "no"],
        default="yes",
        help="Tear down the test environment after running (default: yes)",
    )
    args = parser.parse_args()
    print(f"\nRunning test: {args.test}")
    print(f"Description: {TEST_FUNCTIONS[args.test].__doc__}\n")

    @omero_connect
    def run_e2etest(conn: BlitzGateway = None) -> int:
        try:
            assert conn is not None
            plate_id = e2e_excel_setup(conn)
            print(f"Created plate with ID: {plate_id}")
            plate = conn.getObject("Plate", plate_id)

            TEST_FUNCTIONS[args.test](conn=conn, plate_id=plate_id)

            # Teardown if requested
            if args.teardown == "yes":
                print(f"Cleaning up plate with ID: {plate_id}")
                cleanup_plate(conn, plate)

            return 0
        except Exception as e:  # noqa: BLE001
            # The rich error will have already been displayed
            print(f"Error: {e}")
            return 1

    return run_e2etest()


if __name__ == "__main__":
    sys.exit(main())
