"""RSA-OAEP browser password transport owned by one HTTP process runtime."""

from __future__ import annotations

import base64
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidKey, InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import BaseModel, ConfigDict, SkipValidation

from gewu_core.blocking import run_cpu_task
from gewu_core.file_tasks import FileTaskLane, run_file_task
from gewu_core.http.settings import PasswordTransportSettings

PASSWORD_TRANSPORT_ALGORITHM = "RSA-OAEP-256"
PASSWORD_TRANSPORT_KEY_ID_LENGTH = 16


class _PasswordTransportKeyring(BaseModel):
    """Parsed active and rotation-grace keys installed for request-time use."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    active_key_id: str
    active_private_key: SkipValidation[rsa.RSAPrivateKey]
    active_public_key_pem: str
    private_keys_by_id: SkipValidation[dict[str, rsa.RSAPrivateKey]]


class RsaPasswordTransport:
    """RSA-OAEP implementation of the browser password transport contract."""

    def __init__(self, keyring: _PasswordTransportKeyring | None) -> None:
        self._keyring = keyring

    @classmethod
    async def load(
        cls,
        settings: PasswordTransportSettings,
        project_home: str | Path,
        *,
        required: bool,
    ) -> RsaPasswordTransport:
        """Read and parse a complete keyring outside the event-loop thread."""
        keyring = await run_file_task(
            _build_keyring,
            settings,
            project_home,
            required=required,
            lane=FileTaskLane.INTERACTIVE,
        )
        return cls(keyring)

    def decrypt(self, encrypted_password: str) -> str:
        """Decrypt one browser password envelope."""
        return _decrypt_transport_password(encrypted_password, self._require_keyring())

    async def decrypt_async(self, encrypted_password: str) -> str:
        """Decrypt one browser password envelope through the bounded CPU lane."""
        return await run_cpu_task(
            _decrypt_transport_password,
            encrypted_password,
            self._require_keyring(),
        )

    def encrypt_for_transport(self, password: str) -> str:
        """Encrypt one password for tests and non-browser internal clients."""
        keyring = self._require_keyring()
        ciphertext = keyring.active_private_key.public_key().encrypt(
            password.encode("utf-8"),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        encoded = base64.b64encode(ciphertext).decode("ascii")
        return f"{keyring.active_key_id}.{encoded}"

    def public_key_payload(self) -> dict[str, str]:
        """Return the active browser encryption public key."""
        keyring = self._require_keyring()
        return {
            "algorithm": PASSWORD_TRANSPORT_ALGORITHM,
            "key_id": keyring.active_key_id,
            "public_key_pem": keyring.active_public_key_pem,
        }

    def _require_keyring(self) -> _PasswordTransportKeyring:
        keyring = self._keyring
        if keyring is None:
            raise RuntimeError("password_transport.private_key_path must be configured.")
        return keyring


def _decrypt_transport_password(
    encrypted_password: str,
    keyring: _PasswordTransportKeyring,
) -> str:
    """Decrypt an envelope using one already loaded keyring."""
    try:
        key_id, encoded_ciphertext = encrypted_password.split(".", 1)
        if len(key_id) != PASSWORD_TRANSPORT_KEY_ID_LENGTH:
            raise ValueError("Invalid password transport key id.")
        private_key = keyring.private_keys_by_id.get(key_id)
        if private_key is None:
            raise ValueError("Unknown password transport key id.")
        ciphertext = base64.b64decode(encoded_ciphertext.encode("ascii"), validate=True)
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return plaintext.decode("utf-8")
    except (
        InvalidKey,
        InvalidSignature,
        UnicodeDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        raise ValueError("Invalid encrypted password.") from exc


def _build_keyring(
    settings: PasswordTransportSettings,
    project_home: str | Path,
    *,
    required: bool,
) -> _PasswordTransportKeyring | None:
    """Read, parse and validate one complete keyring before publication."""
    active_path = _normalized_key_path(settings.private_key_path, project_home)
    if not active_path:
        if required:
            raise RuntimeError("password_transport.private_key_path must be configured.")
        return None
    previous_paths = _previous_private_key_paths(
        settings.previous_private_key_paths_json,
        project_home,
    )
    keys = tuple(_load_private_key_file(path) for path in (active_path, *previous_paths))
    key_ids = [_key_id(key) for key in keys]
    if len(key_ids) != len(set(key_ids)):
        raise RuntimeError("Password transport key ring contains duplicate keys.")
    active_key = keys[0]
    public_key_pem = (
        active_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    return _PasswordTransportKeyring(
        active_key_id=key_ids[0],
        active_private_key=active_key,
        active_public_key_pem=public_key_pem,
        private_keys_by_id=dict(zip(key_ids, keys, strict=True)),
    )


def _previous_private_key_paths(
    value: str,
    project_home: str | Path,
) -> tuple[str, ...]:
    """Parse private-key file paths retained during rotation."""
    try:
        parsed: Any = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "password_transport.previous_private_key_paths_json must be JSON."
        ) from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise RuntimeError(
            "password_transport.previous_private_key_paths_json must be a string array."
        )
    return tuple(path for item in parsed if (path := _normalized_key_path(item, project_home)))


def _normalized_key_path(value: str, project_home: str | Path) -> str:
    """Normalize one private-key path relative to the bootstrapped project home."""
    if not value.strip():
        return ""
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = Path(project_home).expanduser() / path
    return str(path.resolve())


def _load_private_key_file(path: str) -> rsa.RSAPrivateKey:
    """Read and parse one configured private-key file."""
    try:
        pem = Path(path).read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Unable to read password transport private key: {path}") from exc
    return _load_private_key(pem)


@lru_cache(maxsize=16)
def _load_private_key(pem: str) -> rsa.RSAPrivateKey:
    """Parse and validate one unencrypted RSA private key."""
    try:
        private_key = serialization.load_pem_private_key(pem.encode("ascii"), password=None)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError("Invalid password transport RSA private key.") from exc
    if not isinstance(private_key, rsa.RSAPrivateKey) or private_key.key_size < 2048:
        raise RuntimeError("Password transport key must be an RSA key of at least 2048 bits.")
    return private_key


def _key_id(private_key: rsa.RSAPrivateKey) -> str:
    """Return the stable public-key fingerprint used in encrypted envelopes."""
    der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()[:PASSWORD_TRANSPORT_KEY_ID_LENGTH]
