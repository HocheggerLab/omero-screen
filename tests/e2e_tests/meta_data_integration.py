import argparse
import logging
import os
import sys

# Clean any existing logger configuration
logger = logging.getLogger("omero")
if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


# Set production environment variables
os.environ["ENV"] = "production"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["LOG_FILE_PATH"] = "/tmp/omero_screen.log"
os.environ["ENABLE_FILE_LOGGING"] = "true"
os.environ["ENABLE_CONSOLE_LOGGING"] = "false"


from omero_screen.metadata_parser import MetadataParser  # noqa: E402
from omero_utils.omero_connect import omero_connect  # noqa: E402
from tests.e2e_tests.excel_file_handling import (  # noqa: E402
    run_excel_file_handling,  # noqa: E402
)

# Clear any existing credentials
if "USERNAME" in os.environ:
    del os.environ["USERNAME"]
if "PASSWORD" in os.environ:
    del os.environ["PASSWORD"]
if "HOST" in os.environ:
    del os.environ["HOST"]
os.environ["USERNAME"] = "root"
os.environ["PASSWORD"] = "omero"
os.environ["HOST"] = "localhost"


def failed_auth_test():
    """Test behavior with invalid credentials"""
    # Set wrong credentials before attempting connection
    if "USERNAME" in os.environ:
        del os.environ["USERNAME"]
    if "HOST" in os.environ:
        del os.environ["HOST"]

    os.environ["USERNAME"] = "wrong_user"
    os.environ["HOST"] = "wrong_host"

    @omero_connect
    def attempt_connection(conn=None):
        parser = MetadataParser(conn, 5000)
        parser._parse_metadata()

    # This should raise an authentication error
    attempt_connection()


@omero_connect
def missing_plate(conn=None):
    """Test basic metadata parsing functionality"""
    parser = MetadataParser(conn, 5000)
    parser._parse_metadata()
    print(parser.well_data)


# Dictionary mapping test names to functions
TEST_FUNCTIONS = {
    "missing_plate": missing_plate,
    "failed_auth": failed_auth_test,
    "excel_file_handling": run_excel_file_handling,
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
