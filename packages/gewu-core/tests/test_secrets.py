"""Stored credential encryption with explicit bootstrap material."""

import pytest

from gewu_core.secrets import (
    ConfiguredJsonSecretCipher,
    JsonSecretCipher,
    StorageEncryptionSettings,
    validate_storage_encryption_configuration,
)


def test_json_secret_cipher_round_trips_without_global_configuration() -> None:
    cipher = JsonSecretCipher("e" * 32)
    ciphertext = cipher.encrypt({"api_key": "secret", "nested": {"enabled": True}})

    assert cipher.decrypt(ciphertext) == {
        "api_key": "secret",
        "nested": {"enabled": True},
    }


def test_json_secret_cipher_fails_closed_for_wrong_key_or_missing_material() -> None:
    ciphertext = JsonSecretCipher("e" * 32).encrypt({"api_key": "secret"})

    with pytest.raises(ValueError, match="Invalid encrypted secret payload"):
        JsonSecretCipher("x" * 32).decrypt(ciphertext)
    with pytest.raises(ValueError, match="must be configured"):
        JsonSecretCipher("  ")


def test_configured_json_secret_cipher_defers_and_names_missing_key_validation() -> None:
    cipher = ConfiguredJsonSecretCipher("  ", setting_name="storage_encryption.key")

    with pytest.raises(RuntimeError, match=r"^storage_encryption\.key must be configured\.$"):
        cipher.encrypt({"secret": "value"})

    configured = ConfiguredJsonSecretCipher(
        "runtime-key",
        setting_name="storage_encryption.key",
    )
    ciphertext = configured.encrypt({"secret": "value"})
    assert configured.decrypt(ciphertext) == {"secret": "value"}


def test_storage_encryption_configuration_requires_a_256_bit_production_key() -> None:
    validate_storage_encryption_configuration(StorageEncryptionSettings(), "dev")

    with pytest.raises(RuntimeError, match="storage_encryption.key must be configured"):
        validate_storage_encryption_configuration(
            StorageEncryptionSettings(),
            "dev",
            required=True,
        )
    validate_storage_encryption_configuration(
        StorageEncryptionSettings(
            key="development-key",
        ),
        "dev",
        required=True,
    )

    with pytest.raises(RuntimeError, match="storage_encryption.key must be configured"):
        validate_storage_encryption_configuration(StorageEncryptionSettings(), "prod")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        validate_storage_encryption_configuration(
            StorageEncryptionSettings(key="short"),
            "prod",
        )

    validate_storage_encryption_configuration(
        StorageEncryptionSettings(key="e" * 32),
        "prod",
    )
