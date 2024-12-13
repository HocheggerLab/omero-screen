import pytest
from omero_utils import omero_connect


# Test for successful connection
def test_omero_connect_success(mock_env, mock_blitzgateway, capsys):
    # Mock a decorated function
    @omero_connect
    def dummy_function(param1, conn=None):
        return param1, conn  # Return the connection object for assertions

    # Call the decorated function
    result_param, result_conn = dummy_function("test_param")

    # Assertions
    assert result_param == "test_param", "Parameter mismatch"
    assert result_conn == mock_blitzgateway, "Connection object mismatch"
    mock_blitzgateway.connect.assert_called_once()  # Ensure connection was established
    mock_blitzgateway.close.assert_called_once()  # Ensure connection was closed

    # Capture and assert print output
    captured = capsys.readouterr()
    assert "Connecting to Omero at at host: test_host" in captured.out
    assert "Closing connection to Omero at host: test_host" in captured.out


def test_omero_connect_failure(mock_env, mock_blitzgateway, capsys):
    # Simulate connection failure
    mock_blitzgateway.connect.side_effect = Exception("Connection failed")
    mock_blitzgateway.isConnected.return_value = (
        False  # Ensure `isConnected` reflects failure
    )

    # Mock a decorated function
    @omero_connect
    def dummy_function(param1, conn=None):
        return (
            param1,
            conn,
        )  # This should not execute due to connection failure

    # Verify that an exception is raised
    with pytest.raises(Exception, match="Connection failed"):
        dummy_function("test_param")

    # Verify `connect` was called and `close` was not
    mock_blitzgateway.connect.assert_called_once()
    mock_blitzgateway.close.assert_not_called()  # Ensure `close` is not called on failure

    # Capture and assert the output
    captured = capsys.readouterr()
    assert "Connecting to Omero at at host: test_host" in captured.out
    assert (
        "Failed to connect to Omero with the following error message: Connection failed"
        in captured.out
    )
    assert (
        "Closing connection" not in captured.out
    )  # Ensure no closing message is printed
