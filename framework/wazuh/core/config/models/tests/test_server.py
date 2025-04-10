from unittest.mock import patch

import pytest
from pydantic import ValidationError
from wazuh.core.config.models.server import (
    DEFAULT_CTI_URL,
    CTIConfig,
    JWTConfig,
    ServerConfig,
)


@pytest.mark.parametrize(
    'init_values, expected',
    [
        ({}, {'update_check': True, 'url': DEFAULT_CTI_URL}),
        ({'update_check': False, 'url': 'www.wazuh.com'}, {'update_check': False, 'url': 'www.wazuh.com'}),
    ],
)
def test_cti_config_default_values(init_values, expected):
    """Check the correct initialization of the `CTIConfig` class."""
    config = CTIConfig(**init_values)

    assert config.update_check == expected['update_check']
    assert config.url == expected['url']


@pytest.mark.parametrize('init_values', [{'private_key': 'private_key_example'}])
@patch('os.path.isfile', return_value=True)
def test_jwt_config_default_values(file_exists_mock, init_values):
    """Check the correct initialization of the `JWTConfig` class."""
    jwt_config = JWTConfig(**init_values)

    assert jwt_config.private_key == init_values['private_key']


@pytest.mark.parametrize(
    'init_values, expected',
    [
        (
            {
                'jwt': {'public_key': 'value', 'private_key': 'value'},
            },
            {
                'update_check': False,
            },
        )
    ],
)
@patch('wazuh.core.config.models.base.ValidateFilePathMixin._validate_file_path')
def test_server_config_default_values(file_path_validation_mock, init_values, expected):
    """Check the correct initialization of the `ServerConfig` class."""
    config = ServerConfig(**init_values)
    assert config.jwt == expected['jwt']
    assert config.update_check == expected['update_check']


def test_server_config_invalid_values():
    """Check the correct behavior of the `ServerConfig` class validations."""
    with pytest.raises(ValidationError):
        _ = ServerConfig(update_check='test')
