from omero_utils.omero_connect import omero_connect


def test_successful_connection():
    @omero_connect
    def check_connection(conn):
        # Just check if we can get the server version, which doesn't require any data
        return conn.getSession()

    server_version = check_connection()

    assert server_version is not None, (
        "Failed to get server version - connection may not be established"
    )


# def test_connection_failure(capsys, clean_env):
#     # Set wrong credentials
#     os.environ["USERNAME"] = "wrong_user"
#     os.environ["PASSWORD"] = "wrong_password"
#     os.environ["HOST"] = "localhost"  # Keep the host the same

#     @omero_connect
#     def connect_plate(conn):
#         return conn.getObject("Plate", 53)

#     with pytest.raises(Exception):  # noqa: B017
#         connect_plate()

#     # Capture the stdout and stderr
#     captured = capsys.readouterr()
#     assert "Failed to connect to Omero" in captured.out, (
#         "Expected error message not found in stdout"
#     )
#     )
#     )
#     )


# import os

# import pytest
# from dotenv import load_dotenv
# from omero_utils.omero_connect import omero_connect

# from omero_screen.config import set_env_vars


# @pytest.fixture(autouse=True)
# def setup_environment():
#     """Fixture to automatically set up environment variables before each test"""
#     dotenv_path = set_env_vars()
#     load_dotenv(dotenv_path=dotenv_path)
#     # Ensure we have the minimum required environment variables
#     assert (
#         os.getenv("USERNAME") is not None
#     ), "USERNAME environment variable not set"
#     assert (
#         os.getenv("PASSWORD") is not None
#     ), "PASSWORD environment variable not set"
#     assert os.getenv("HOST") is not None, "HOST environment variable not set"


# def test_set_env_vars_local(clean_env):
#     dotenv_path = set_env_vars()
#     load_dotenv(dotenv_path=dotenv_path)

#     username = os.getenv("USERNAME")
#     password = os.getenv("PASSWORD")

#     assert username == "root", "Username is not correct"
#     assert password == "omero", "Password is not correct"


# def test_successful_connection(clean_env):
#     dotenv_path = set_env_vars()
#     load_dotenv(dotenv_path=dotenv_path)
#     @omero_connect
#     def check_connection(conn):
#         # Just check if we can get the server version, which doesn't require any data
#         return conn.getSession()

#     server_version = check_connection()

#     assert (
#         server_version is not None
#     ), "Failed to get server version - connection may not be established"


# # def test_connection_failure(capsys, clean_env):
# #     # Set wrong credentials
# #     os.environ["USERNAME"] = "wrong_user"
# #     os.environ["PASSWORD"] = "wrong_password"
# #     os.environ["HOST"] = "localhost"  # Keep the host the same

# #     @omero_connect
# #     def connect_plate(conn):
# #         return conn.getObject("Plate", 53)

# #     with pytest.raises(Exception):  # noqa: B017
# #         connect_plate()

# #     # Capture the stdout and stderr
# #     captured = capsys.readouterr()
# #     assert "Failed to connect to Omero" in captured.out, (
# #         "Expected error message not found in stdout"
# #     )
# #     )
# #     )
# #     )


# # import pytest
# # from omero_utils import omero_connect


# # # Test for successful connection
# # def test_omero_connect_success(mock_env, mock_blitzgateway, capsys):
# #     # Mock a decorated function
# #     @omero_connect
# #     def dummy_function(param1, conn=None):
# #         return param1, conn  # Return the connection object for assertions

# #     # Call the decorated function
# #     result_param, result_conn = dummy_function("test_param")

# #     # Assertions
# #     assert result_param == "test_param", "Parameter mismatch"
# #     assert result_conn == mock_blitzgateway, "Connection object mismatch"
# #     mock_blitzgateway.connect.assert_called_once()  # Ensure connection was established
# #     mock_blitzgateway.close.assert_called_once()  # Ensure connection was closed

# #     # Capture and assert print output
# #     captured = capsys.readouterr()
# #     assert "Connecting to Omero at at host: test_host" in captured.out
# #     assert "Closing connection to Omero at host: test_host" in captured.out


# # def test_omero_connect_failure(mock_env, mock_blitzgateway, capsys):
# #     # Simulate connection failure
# #     mock_blitzgateway.connect.side_effect = Exception("Connection failed")
# #     mock_blitzgateway.isConnected.return_value = (
# #         False  # Ensure `isConnected` reflects failure
# #     )

# #     # Mock a decorated function
# #     @omero_connect
# #     def dummy_function(param1, conn=None):
# #         return (
# #             param1,
# #             conn,
# #         )  # This should not execute due to connection failure

# #     # Verify that an exception is raised
# #     with pytest.raises(Exception, match="Connection failed"):
# #         dummy_function("test_param")

# #     # Verify `connect` was called and `close` was not
# #     mock_blitzgateway.connect.assert_called_once()
# #     mock_blitzgateway.close.assert_not_called()  # Ensure `close` is not called on failure

# #     # Capture and assert the output
# #     captured = capsys.readouterr()
# #     assert "Connecting to Omero at at host: test_host" in captured.out
# #     assert (
# #         "Failed to connect to Omero with the following error message: Connection failed"
# #         in captured.out
# #     )
# #     assert (
# #         "Closing connection" not in captured.out
# #     )  # Ensure no closing message is printed
