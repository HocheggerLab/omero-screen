"""
This script is used to run the integration tests.
It will create a plate, run the test, and then tear down the plate.
The test can be chosen from the command line.
The teardown can be chosen with the --teardown flag with yes or no as options.
The default is yes.
Run the script with omero-integration-test test_name
New tests should be added to the TEST_FUNCTIONS dictionary.
"""

import argparse
import os

# Set up environment
os.environ["ENV"] = "e2etest"
import sys
from typing import Optional

from omero.gateway import BlitzGateway
from omero_utils.omero_connect import omero_connect
from omero_utils.omero_plate import cleanup_plate

from tests.e2e_tests.e2e_connection import (
    failed_connection,
    successful_connection,
)
from tests.e2e_tests.e2e_excel import (
    missing_plate,
    run_plate_with_correct_excel,
)
from tests.e2e_tests.e2e_setup import e2e_excel_setup

# Set up output redirection for SLURM mode before any other imports
parser = argparse.ArgumentParser(description="Run metadata integration tests")
parser.add_argument("test", help="Test to run")
parser.add_argument(
    "-t",
    "--teardown",
    choices=["yes", "no"],
    default="yes",
    help="Tear down the test environment after running (default: yes)",
)
parser.add_argument(
    "-c",
    "--connection",
    action="store_true",
    help="Run connection test with omero_connect decorator",
)
args, _ = (
    parser.parse_known_args()
)  # Use known_args to avoid conflicts with other parsers


# Now import everything else


# Dictionary mapping test names to functions
TEST_FUNCTIONS = {
    "failed_connection": failed_connection,
    "successful_connection": successful_connection,
    "e2e_excel": run_plate_with_correct_excel,
    "missing_plate": missing_plate,
}


def main() -> None:
    """Main entry point for the integration tests"""
    print(f"\nRunning test: {args.test}")
    print(f"Description: {TEST_FUNCTIONS[args.test].__doc__}\n")

    if args.connection:
        TEST_FUNCTIONS[args.test]()
    else:

        @omero_connect
        def run_e2etest(conn: Optional[BlitzGateway] = None) -> None:
            assert conn is not None
            plate_id = e2e_excel_setup(conn)
            plate = conn.getObject("Plate", plate_id)

            # Run the test
            TEST_FUNCTIONS[args.test](conn=conn, plate_id=plate_id)

            # Teardown if requested
            if args.teardown == "yes":
                cleanup_plate(conn, plate)

        run_e2etest()


if __name__ == "__main__":
    sys.exit(main())
