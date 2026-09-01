"""Browser password transport key loading and rotation behavior."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import gewu_core.http.password_transport as password_transport_module
from gewu_core.http.password_transport import RsaPasswordTransport
from gewu_core.http.settings import PasswordTransportSettings


async def test_password_transport_round_trips_and_exposes_algorithm(tmp_path: Path) -> None:
    transport = await _load_transport(tmp_path / "active.pem")

    encrypted = transport.encrypt_for_transport("s3cret")
    payload = transport.public_key_payload()

    assert encrypted != "s3cret"
    assert transport.decrypt(encrypted) == "s3cret"
    assert payload["algorithm"] == "RSA-OAEP-256"
    assert payload["key_id"] == encrypted.split(".", 1)[0]
    assert "BEGIN PUBLIC KEY" in payload["public_key_pem"]


async def test_loaded_transport_does_not_reopen_key_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = await _load_transport(tmp_path / "active.pem")

    def reject_read(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("request-time key file read")

    monkeypatch.setattr(Path, "read_text", reject_read)
    encrypted = transport.encrypt_for_transport("s3cret")

    assert transport.decrypt(encrypted) == "s3cret"
    assert transport.public_key_payload()["key_id"]


async def test_password_decryption_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = await _load_transport(tmp_path / "active.pem")
    encrypted = transport.encrypt_for_transport("s3cret")
    original_decrypt = password_transport_module._decrypt_transport_password
    release = threading.Event()

    def slow_decrypt(envelope: str, keyring: object) -> str:
        release.wait(timeout=0.2)
        return original_decrypt(envelope, keyring)  # type: ignore[arg-type]

    monkeypatch.setattr(password_transport_module, "_decrypt_transport_password", slow_decrypt)
    started = time.perf_counter()
    task = asyncio.create_task(transport.decrypt_async(encrypted))
    await asyncio.sleep(0.02)
    heartbeat_elapsed = time.perf_counter() - started
    release.set()

    assert await task == "s3cret"
    assert heartbeat_elapsed < 0.1


async def test_password_transport_keeps_previous_key_during_rotation(tmp_path: Path) -> None:
    old_path = tmp_path / "old.pem"
    old_transport = await _load_transport(old_path)
    encrypted = old_transport.encrypt_for_transport("s3cret")
    new_path = tmp_path / "new.pem"
    _write_private_key(new_path)

    rotated = await RsaPasswordTransport.load(
        PasswordTransportSettings(
            private_key_path=str(new_path),
            previous_private_key_paths_json=json.dumps([str(old_path)]),
        ),
        tmp_path,
        required=True,
    )

    assert rotated.public_key_payload()["key_id"] != encrypted.split(".", 1)[0]
    assert rotated.decrypt(encrypted) == "s3cret"


async def test_password_transport_rejects_unknown_or_unkeyed_ciphertext(tmp_path: Path) -> None:
    old_transport = await _load_transport(tmp_path / "old.pem")
    encrypted = old_transport.encrypt_for_transport("s3cret")
    replacement = await _load_transport(tmp_path / "replacement.pem")

    with pytest.raises(ValueError, match="Invalid encrypted password"):
        replacement.decrypt(encrypted)
    with pytest.raises(ValueError, match="Invalid encrypted password"):
        replacement.decrypt("Zm9v")


async def test_production_requires_key_and_rotation_ring_is_strict(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="private_key_path must be configured"):
        await RsaPasswordTransport.load(
            PasswordTransportSettings(),
            tmp_path,
            required=True,
        )

    active_path = tmp_path / "active.pem"
    _write_private_key(active_path)
    with pytest.raises(RuntimeError, match="must be a string array"):
        await RsaPasswordTransport.load(
            PasswordTransportSettings(
                private_key_path=str(active_path),
                previous_private_key_paths_json='{"not": "a list"}',
            ),
            tmp_path,
            required=True,
        )


async def test_relative_key_paths_resolve_from_bootstrap_project_home(tmp_path: Path) -> None:
    project_home = tmp_path / "project"
    active_path = project_home / "conf" / "active.pem"
    previous_path = project_home / "conf" / "previous.pem"
    active_path.parent.mkdir(parents=True)
    _write_private_key(active_path)
    _write_private_key(previous_path)

    transport = await RsaPasswordTransport.load(
        PasswordTransportSettings(
            private_key_path="conf/active.pem",
            previous_private_key_paths_json=json.dumps(["conf/previous.pem"]),
        ),
        project_home,
        required=True,
    )

    assert transport.public_key_payload()["key_id"]


async def _load_transport(path: Path) -> RsaPasswordTransport:
    _write_private_key(path)
    return await RsaPasswordTransport.load(
        PasswordTransportSettings(private_key_path=str(path)),
        path.parent,
        required=True,
    )


def _write_private_key(path: Path) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path.write_text(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("ascii"),
        encoding="ascii",
    )
