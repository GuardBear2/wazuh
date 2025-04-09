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

                    from wazuh.core.exception import (
                        WazuhError,
                        WazuhInternalError,
                        WazuhPermissionError,
                        WazuhResourceNotFound,
                    )
                    from wazuh.core.results import WazuhResult
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


def test_get_manager_status():
    """Check that get_manager_status function returns the manager status.

    For this test, the status can be stopped or failed.
    """
    called = 0

    def exist_mock(path):
        if '.failed' in path and called == 0:
            return True
        elif '.restart' in path and called == 1:
            return True
        elif '.start' in path and called == 2:
            return True
        elif '/proc' in path and called == 3:
            return True
        else:
            return False

    status = utils.get_manager_status()
    for value in status.values():
        assert value == 'stopped'

    with patch('wazuh.core.server.utils.glob', return_value=['ossec-0.pid']):
        with patch('re.match', return_value='None'):
            status = utils.get_manager_status()
            for value in status.values():
                assert value == 'failed'

        with patch('wazuh.core.server.utils.os.path.exists', side_effect=exist_mock):
            status = utils.get_manager_status()
            for value in status.values():
                assert value == 'failed'

            called += 1
            status = utils.get_manager_status()
            for value in status.values():
                assert value == 'restarting'

            called += 1
            status = utils.get_manager_status()
            for value in status.values():
                assert value == 'starting'

            called += 1
            status = utils.get_manager_status()
            for value in status.values():
                assert value == 'running'


@pytest.mark.parametrize('exc', [PermissionError, FileNotFoundError])
@patch('os.stat')
def test_get_manager_status_ko(mock_stat, exc):
    """Check that get_manager_status function correctly handles expected exceptions."""
    mock_stat.side_effect = exc
    with pytest.raises(WazuhInternalError, match='.* 1913 .*'):
        utils.get_manager_status()


def test_manager_restart():
    """Verify that manager_restart send to the manager the restart request."""
    with patch('wazuh.core.server.utils.open', side_effect=None):
        with patch('fcntl.lockf', side_effect=None):
            with pytest.raises(WazuhInternalError, match='.* 1901 .*'):
                utils.manager_restart()

            with patch('os.path.exists', return_value=True):
                with pytest.raises(WazuhInternalError, match='.* 1902 .*'):
                    utils.manager_restart()

                with patch('socket.socket.connect', side_effect=None):
                    with pytest.raises(WazuhInternalError, match='.* 1014 .*'):
                        utils.manager_restart()

                    with patch('socket.socket.send', side_effect=None):
                        status = utils.manager_restart()
                        assert WazuhResult({'message': 'Restart request sent'}) == status


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


@pytest.mark.parametrize(
    'result',
    [
        WazuhError(6001),
        WazuhInternalError(1000),
        WazuhPermissionError(4000),
        WazuhResourceNotFound(1710),
        'value',
        1,
        False,
        {'key': 'value'},
    ],
)
def test_raise_if_exc(result):
    """Check that raise_if_exc raises an exception if the result is one."""
    if isinstance(result, Exception):
        with pytest.raises(Exception):
            utils.raise_if_exc(result)
    else:
        utils.raise_if_exc(result)
