# Copyright (C) 2015, Wazuh Inc.
# Created by Wazuh, Inc. <info@wazuh.com>.
# This program is a free software; you can redistribute it and/or modify it under the terms of GPLv2

import hashlib
import json
import os
from copy import deepcopy
from unittest.mock import ANY, MagicMock, call, patch

from connexion.exceptions import Unauthorized

with patch('wazuh.core.common.wazuh_uid'):
    with patch('wazuh.core.common.wazuh_gid'):
        from wazuh.core.indexer.models.rbac import User
        from wazuh.core.results import WazuhResult

import pytest

with patch('wazuh.core.common.wazuh_uid'):
    with patch('wazuh.core.common.wazuh_gid'):
        from server_management_api import authentication


test_path = os.path.dirname(os.path.realpath(__file__))
test_data_path = os.path.join(test_path, 'data')

security_conf = WazuhResult({'auth_token_exp_timeout': 900, 'rbac_mode': 'black'})
decoded_payload = {
    'iss': 'wazuh',
    'aud': 'Wazuh API REST',
    'nbf': 0,
    'exp': security_conf['auth_token_exp_timeout'],
    'sub': '001',
    'rbac_policies': {'value': 'test', 'rbac_mode': security_conf['rbac_mode']},
    'rbac_roles': [1],
    'run_as': False,
}

original_payload = {
    'iss': 'wazuh',
    'aud': 'Wazuh API REST',
    'nbf': 0,
    'exp': security_conf['auth_token_exp_timeout'],
    'sub': '001',
    'run_as': False,
    'rbac_roles': [1],
    'rbac_mode': security_conf['rbac_mode'],
}


@patch('server_management_api.authentication.rbac_manager')
async def test_check_user_master(rbac_manager_var_mock):
    """Validate that the `check_user_master` function works as expected."""
    rbac_manager_mock = MagicMock()
    rbac_manager_mock.get_users_by_name.return_value = User(name='test_user', raw_password='test_pass')
    rbac_manager_var_mock.get.return_value = rbac_manager_mock
    result = await authentication.check_user_master('test_user', 'test_pass')
    assert result == {'result': True}


@patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.__init__', return_value=None)
@patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.execute_function', side_effect=None)
@patch('server_management_api.authentication.raise_if_exc', side_effect=None)
async def test_check_user(mock_raise_if_exc, mock_execute_function, mock_dapi):
    """Verify if result is as expected."""
    mock_request = MagicMock()
    mock_state = MagicMock()
    mock_request.state = mock_state
    result = authentication.check_user('test_user', 'test_pass', request=mock_request)

    assert result == {'sub': 'test_user', 'active': True}, 'Result is not as expected'
    mock_dapi.assert_called_once_with(
        f=ANY,
        f_kwargs={'name': 'test_user', 'password': 'test_pass'},
        is_async=True,
        wait_for_complete=False,
        logger=ANY,
        rbac_manager=mock_state.rbac_manager,
    )
    mock_execute_function.assert_called_once_with()
    mock_raise_if_exc.assert_called_once()


def test_get_security_conf():
    """Check that returned object is as expected."""
    result = authentication.get_security_conf()
    assert isinstance(result, dict)
    assert all(x in result.keys() for x in ('auth_token_exp_timeout', 'rbac_mode'))


@pytest.mark.asyncio
@pytest.mark.parametrize('auth_context', [{'name': 'initial_auth'}, None])
@patch('server_management_api.authentication.jwt.encode', return_value='test_token')
@patch(
    'server_management_api.authentication.get_keypair',
    return_value=('-----BEGIN PRIVATE KEY-----', '-----BEGIN PUBLIC KEY-----'),
)
@patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.__init__', return_value=None)
@patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.execute_function', side_effect=None)
@patch('server_management_api.authentication.raise_if_exc', side_effect=None)
async def test_generate_token(
    mock_raise_if_exc, mock_execute_function, mock_dapi, mock_get_keypair, mock_encode, auth_context
):
    """Verify if result is as expected."""

    class NewDatetime:
        def timestamp(self) -> float:
            return 0

    mock_raise_if_exc.return_value = security_conf
    with patch('server_management_api.authentication.core_utils.get_utc_now', return_value=NewDatetime()):
        result = authentication.generate_token(user_id='001', data={'roles': [1]}, auth_context=auth_context)
    assert result == 'test_token', 'Result is not as expected'

    # Check all functions are called with expected params
    mock_dapi.assert_called_once_with(
        f=ANY, request_type='local_master', is_async=False, wait_for_complete=False, logger=ANY
    )
    mock_execute_function.assert_called_once_with()
    mock_raise_if_exc.assert_called_once()
    mock_get_keypair.assert_called_once()
    expected_payload = original_payload | (
        {
            'hash_auth_context': hashlib.blake2b(json.dumps(auth_context).encode(), digest_size=16).hexdigest(),
            'run_as': True,
        }
        if auth_context is not None
        else {}
    )
    mock_encode.assert_called_once_with(expected_payload, '-----BEGIN PRIVATE KEY-----', algorithm='RS256')


@patch('server_management_api.authentication.rbac_manager')
async def test_check_token(rbac_manager_var_mock):
    """Validate that the `check_token` function works as expected."""
    rbac_manager_mock = MagicMock()
    rbac_manager_mock.get_users_by_name.return_value = User(name='test_user', raw_password='test_pass')
    rbac_manager_var_mock.get.return_value = rbac_manager_mock
    result = await authentication.check_token(username='wazuh_user', roles=tuple(), token_nbf_time=3600, run_as=False)
    assert result == {'valid': ANY, 'policies': ANY}


@pytest.mark.asyncio
@patch('server_management_api.authentication.jwt.decode')
@patch(
    'server_management_api.authentication.get_keypair',
    return_value=('-----BEGIN PRIVATE KEY-----', '-----BEGIN PUBLIC KEY-----'),
)
@patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.__init__', return_value=None)
@patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.execute_function', return_value=True)
@patch('server_management_api.authentication.raise_if_exc', side_effect=None)
async def test_decode_token(mock_raise_if_exc, mock_execute_function, mock_dapi, mock_get_keypair, mock_decode):
    """Validate that the `decode_token` function works as expected."""
    mock_decode.return_value = deepcopy(original_payload)
    mock_raise_if_exc.side_effect = [
        WazuhResult({'valid': True, 'policies': {'value': 'test'}}),
        WazuhResult(security_conf),
    ]
    mock_request = MagicMock()
    mock_state = MagicMock()
    mock_request.state = mock_state

    result = authentication.decode_token('test_token', request=mock_request)
    assert result == decoded_payload

    # Check all functions are called with expected params
    calls = [
        call(
            f=ANY,
            f_kwargs={
                'username': original_payload['sub'],
                'token_nbf_time': original_payload['nbf'],
                'run_as': False,
                'roles': tuple(original_payload['rbac_roles']),
            },
                is_async=True,
            wait_for_complete=False,
            logger=ANY,
            rbac_manager=mock_state.rbac_manager,
        ),
        call(f=ANY, request_type='local_master', is_async=False, wait_for_complete=False, logger=ANY),
    ]
    mock_dapi.assert_has_calls(calls)
    mock_get_keypair.assert_called_once()
    mock_decode.assert_called_once_with(
        'test_token', '-----BEGIN PUBLIC KEY-----', algorithms=['RS256'], audience='Wazuh API REST'
    )
    assert mock_execute_function.call_count == 2
    assert mock_raise_if_exc.call_count == 2


@pytest.mark.asyncio
@patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.execute_function', side_effect=None)
@patch('server_management_api.authentication.raise_if_exc', side_effect=None)
@patch(
    'server_management_api.authentication.get_keypair',
    return_value=('-----BEGIN PRIVATE KEY-----', '-----BEGIN PUBLIC KEY-----'),
)
async def test_decode_token_ko(mock_get_keypair, mock_raise_if_exc, mock_execute_function):
    """Assert exceptions are handled as expected inside decode_token()."""
    mock_request = MagicMock()
    mock_state = MagicMock()
    mock_request.state = mock_state

    with pytest.raises(Unauthorized):
        authentication.decode_token(token='test_token', request=mock_request)

    with patch('server_management_api.authentication.jwt.decode') as mock_decode:
        with patch(
            'server_management_api.authentication.get_keypair',
            return_value=('-----BEGIN PRIVATE KEY-----', '-----BEGIN PUBLIC KEY-----'),
        ):
            with patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.__init__', return_value=None):
                with patch('wazuh.core.cluster.dapi.dapi.DistributedAPI.execute_function'):
                    with patch('server_management_api.authentication.raise_if_exc') as mock_raise_if_exc:
                        mock_decode.return_value = deepcopy(original_payload)

                        with pytest.raises(Unauthorized):
                            mock_raise_if_exc.side_effect = [WazuhResult({'valid': False})]
                            authentication.decode_token(token='test_token', request=mock_request)

                        with pytest.raises(Unauthorized):
                            mock_raise_if_exc.side_effect = [
                                WazuhResult({'valid': True, 'policies': {'value': 'test'}}),
                                WazuhResult({'auth_token_exp_timeout': 900, 'rbac_mode': 'white'}),
                            ]
                            authentication.decode_token(token='test_token', request=mock_request)
