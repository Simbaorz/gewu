"""Explicit-key encryption for JSON-compatible stored payloads."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from pydantic import Field

from gewu_core.config import SettingsModel

MIN_PRODUCTION_ENCRYPTION_KEY_BYTES = 32


class StorageEncryptionSettings(SettingsModel):
    """Encryption key for protected values stored by one process."""

    key: str = Field(default="", exclude=True, repr=False)


def require_storage_encryption_key(settings: StorageEncryptionSettings) -> str:
    """Return configured storage key material or fail closed."""

    key = settings.key.strip()
    if not key:
        raise RuntimeError("storage_encryption.key must be configured.")
    return key


def validate_storage_encryption_configuration(
    settings: StorageEncryptionSettings,
    mode: str,
    *,
    required: bool = False,
) -> None:
    """Require key material when encryption is enabled and strengthen it in production."""

    production = mode.strip().lower() == "prod"
    if not production and not required:
        return
    key = require_storage_encryption_key(settings)
    if production and len(key.encode("utf-8")) < MIN_PRODUCTION_ENCRYPTION_KEY_BYTES:
        raise RuntimeError(
            "storage_encryption.key must contain at least "
            f"{MIN_PRODUCTION_ENCRYPTION_KEY_BYTES} bytes in production."
        )


class JsonSecretCipher:
    """Encrypt and decrypt deterministic JSON serialization with one supplied key."""

    def __init__(self, key_material: str) -> None:
        material = key_material.strip()
        if not material:
            raise ValueError("Credential encryption key must be configured.")
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(digest))

    def encrypt(self, payload: dict[str, Any]) -> str:
        """Encrypt one JSON-compatible mapping."""

        content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return self._fernet.encrypt(content.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> dict[str, Any]:
        """Decrypt one mapping or reject malformed and wrong-key ciphertext."""

        if not ciphertext:
            return {}
        try:
            content = self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
            payload = json.loads(content)
        except (InvalidToken, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Invalid encrypted secret payload.") from exc
        return payload if isinstance(payload, dict) else {}


class ConfiguredJsonSecretCipher:
    """Resolve bootstrap-provided key material only when encryption is needed."""

    def __init__(self, key_material: str, *, setting_name: str) -> None:
        normalized_name = setting_name.strip()
        if not normalized_name:
            raise ValueError("Secret setting name must be configured.")
        self._key_material = key_material
        self._setting_name = normalized_name

    def encrypt(self, payload: dict[str, Any]) -> str:
        """Encrypt one mapping or fail closed when the configured key is absent."""

        return self._cipher().encrypt(payload)

    def decrypt(self, ciphertext: str) -> dict[str, Any]:
        """Decrypt one mapping or fail closed when the configured key is absent."""

        return self._cipher().decrypt(ciphertext)

    def _cipher(self) -> JsonSecretCipher:
        if not self._key_material.strip():
            raise RuntimeError(f"{self._setting_name} must be configured.")
        return JsonSecretCipher(self._key_material)
