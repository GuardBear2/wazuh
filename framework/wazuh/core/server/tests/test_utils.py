import pathlib
import socket
import sys
from unittest.mock import MagicMock, patch

import pytest
from wazuh.core.config.client import CentralizedConfig
from wazuh.core.config.models.server import ValidateFilePathMixin
from wazuh.core.server.tests.conftest import get_default_configuration

with patch('wazuh.core.common.getgrnam'):
    with patch('wazuh.core.common.getpwnam'):
        with patch('wazuh.core.common.wazuh_uid'):
            with patch('wazuh.core.common.wazuh_gid'):
                with patch.object(ValidateFilePathMixin, '_validate_file_path', return_value=None):
                    default_config = get_default_configuration()
                    CentralizedConfig._config = default_config
                    sys.modules['wazuh.rbac.orm'] = MagicMock()

                    from wazuh.core.server import utils


def test_ping_unix_socket_file_does_not_exist():
    """Verify ping_unix_socket returns False when the socket file does not exist."""
    with patch('pathlib.Path.exists', return_value=False):
        assert not utils.ping_unix_socket(pathlib.Path('/tmp/nonexistent_socket'))


@pytest.mark.parametrize('timeout', (None, 10))
def test_ping_unix_socket_successful(timeout: int | None):
    """Verify a successful connection to the UNIX socket."""
    with patch('pathlib.Path.exists', return_value=True), patch('socket.socket') as mock_socket:
        mock_client = MagicMock()
        mock_socket.return_value = mock_client
        socket_path = pathlib.Path('/tmp/existing_socket')

        assert utils.ping_unix_socket(socket_path, timeout) is True
        mock_client.settimeout.assert_called_once_with(timeout)
        mock_client.connect.assert_called_once_with(str(socket_path))
        mock_client.close.assert_called_once()


def test_ping_unix_socket_connection_timeout():
    """Verify ping_unix_socket returns False when the connection to the UNIX socket times out."""
    with patch('pathlib.Path.exists', return_value=True), patch('socket.socket') as mock_socket:
        mock_client = MagicMock()
        mock_client.connect.side_effect = socket.timeout
        mock_socket.return_value = mock_client
        socket_path = pathlib.Path('/tmp/existing_socket')

        assert not utils.ping_unix_socket(socket_path)
        mock_client.connect.assert_called_once_with(str(socket_path))


def test_ping_unix_socket_error():
    """Verify ping_unix_socket returns False when there is a generic socket error."""
    with patch('pathlib.Path.exists', return_value=True), patch('socket.socket') as mock_socket:
        mock_client = MagicMock()
        mock_client.connect.side_effect = socket.error('Test error')
        mock_socket.return_value = mock_client
        socket_path = pathlib.Path('/tmp/existing_socket')

        assert not utils.ping_unix_socket(socket_path)
        mock_client.connect.assert_called_once_with(str(socket_path))


def test_server_filter():
    """Verify that ServerFilter adds server related information into the logs."""
    server_filter = utils.ServerFilter(tag='Server', subtag='config')
    record = utils.ServerFilter(tag='Testing', subtag='config')
    record.update_tag(new_tag='Testing_tag')
    record.update_subtag(new_subtag='Testing_subtag')

    assert server_filter.filter(record=record)


def test_server_logger():
    """Verify that ServerLogger defines the logger used by wazuh-server."""
    server_logger = utils.ServerLogger(
        tag='%(asctime)s %(levelname)s: [%(tag)s] [%(subtag)s] %(message)s', debug_level=1
    )
    server_logger.setup_logger()
