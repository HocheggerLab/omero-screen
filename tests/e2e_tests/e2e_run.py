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

from tests.e2e_tests.e2e_connection import (
    failed_connection,
    successful_connection,
)
from tests.e2e_tests.e2e_excel import (
    missing_plate,
    run_plate_multierror,
    run_plate_noDAPI,
    run_plate_with_correct_excel,
    run_plate_wrongwell,
)
from tests.e2e_tests.e2e_pixelsize import run_pixel_size_test

# Set up output redirection for SLURM mode before any other imports
parser = argparse.ArgumentParser(description="Run metadata integration tests")
parser.add_argument("test", help="Test to run")
parser.add_argument(
    "-t",
    "--teardown",
    choices=[True, False],
    default=True,
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
    "noDAPI": run_plate_noDAPI,
    "wrongwell": run_plate_wrongwell,
    "multierror": run_plate_multierror,
    "pixel_size": run_pixel_size_test,
}


def main() -> int:
    """Main entry point for the integration tests"""
    if args.test not in TEST_FUNCTIONS:
        print(f"\nERROR: Unknown test: {args.test}")
        print("Choose from: ", list(TEST_FUNCTIONS.keys()))
        return 1

    print(f"\nRunning test: {args.test}")
    print(f"Description: {TEST_FUNCTIONS[args.test].__doc__}\n")

    if args.connection:
        TEST_FUNCTIONS[args.test]()
    else:

        @omero_connect
        def run_e2etest(conn: Optional[BlitzGateway] = None) -> None:
            assert conn is not None
            TEST_FUNCTIONS[args.test](conn=conn, teardown=args.teardown)

        run_e2etest()

    return 0


if __name__ == "__main__":
    sys.exit(main())
