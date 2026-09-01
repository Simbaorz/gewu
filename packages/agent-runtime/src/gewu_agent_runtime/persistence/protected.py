"""Transparent protected-message persistence shared by Runtime stores."""

from __future__ import annotations

from typing import Any, Protocol

from gewu_agent_runtime.domain import (
    ConversationCompaction,
    ConversationMessage,
    NewConversationMessage,
    ProtectedMessageBody,
)

PROTECTED_MESSAGE_ENVELOPE_KEY = "_gewu_protected_message"
PLAINTEXT_ENCRYPTION_VERSION = 0
FERNET_JSON_ENCRYPTION_VERSION = 1


class ProtectedPayloadCipher(Protocol):
    """Encrypt and decrypt one JSON-compatible protected payload."""

    def encrypt(self, payload: dict[str, Any]) -> str:
        """Return opaque authenticated ciphertext."""

    def decrypt(self, ciphertext: str) -> dict[str, Any]:
        """Return authenticated cleartext or fail closed."""


def stored_message(
    message: NewConversationMessage,
    *,
    conversation_id: str,
    sequence: int,
    cipher: ProtectedPayloadCipher | None,
    encrypt: bool,
) -> ConversationMessage:
    """Build one persisted message using the selected protection mode."""

    values = message.model_dump()
    protected = message.protected_body
    if protected is not None:
        if encrypt:
            if cipher is None:
                raise RuntimeError("Protected Runtime message encryption is not configured.")
            values["content"] = ""
            values["payload"] = {
                PROTECTED_MESSAGE_ENVELOPE_KEY: {
                    "version": FERNET_JSON_ENCRYPTION_VERSION,
                    "ciphertext": cipher.encrypt(protected.model_dump(mode="json")),
                }
            }
            values["body_encryption_version"] = FERNET_JSON_ENCRYPTION_VERSION
        else:
            values["content"] = protected.content
            values["payload"] = protected.payload
    return ConversationMessage(
        **values,
        conversation_id=conversation_id,
        sequence=sequence,
    )


def hydrated_message(
    message: ConversationMessage,
    cipher: ProtectedPayloadCipher | None,
) -> ConversationMessage:
    """Restore one protected body while keeping its ciphertext out of projections."""

    if message.body_encryption_version == PLAINTEXT_ENCRYPTION_VERSION:
        return message.model_copy(deep=True)
    if message.body_encryption_version != FERNET_JSON_ENCRYPTION_VERSION:
        raise RuntimeError("Protected Runtime message version is unsupported.")
    envelope = message.payload.get(PROTECTED_MESSAGE_ENVELOPE_KEY)
    if not isinstance(envelope, dict):
        raise RuntimeError("Protected Runtime message envelope is malformed.")
    if envelope.get("version") != FERNET_JSON_ENCRYPTION_VERSION:
        raise RuntimeError("Protected Runtime message version is unsupported.")
    ciphertext = envelope.get("ciphertext")
    if not isinstance(ciphertext, str) or not ciphertext or cipher is None:
        raise RuntimeError("Protected Runtime message cannot be decrypted.")
    body = ProtectedMessageBody.model_validate(cipher.decrypt(ciphertext))
    return message.model_copy(
        update={
            "content": body.content,
            "payload": body.payload,
            "body_encryption_version": PLAINTEXT_ENCRYPTION_VERSION,
        },
        deep=True,
    )


def stored_compaction(
    compaction: ConversationCompaction,
    cipher: ProtectedPayloadCipher | None,
    *,
    encrypt: bool,
) -> ConversationCompaction:
    """Build one stored summary using the selected protection mode."""

    if not encrypt:
        return compaction.model_copy(
            update={"summary_encryption_version": PLAINTEXT_ENCRYPTION_VERSION},
            deep=True,
        )
    if cipher is None:
        raise RuntimeError("Runtime compaction encryption is not configured.")
    ciphertext = cipher.encrypt({"summary": compaction.summary})
    return compaction.model_copy(
        update={
            "summary": ciphertext,
            "summary_encryption_version": FERNET_JSON_ENCRYPTION_VERSION,
        },
        deep=True,
    )


def hydrated_compaction(
    compaction: ConversationCompaction,
    cipher: ProtectedPayloadCipher | None,
) -> ConversationCompaction:
    """Restore an encrypted summary while accepting plain summaries."""

    if compaction.summary_encryption_version == PLAINTEXT_ENCRYPTION_VERSION:
        return compaction.model_copy(deep=True)
    if compaction.summary_encryption_version != FERNET_JSON_ENCRYPTION_VERSION:
        raise RuntimeError("Protected Runtime compaction version is unsupported.")
    if cipher is None:
        raise RuntimeError("Protected Runtime compaction cannot be decrypted.")
    payload = cipher.decrypt(compaction.summary)
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary:
        raise RuntimeError("Protected Runtime compaction envelope is malformed.")
    return compaction.model_copy(
        update={
            "summary": summary,
            "summary_encryption_version": PLAINTEXT_ENCRYPTION_VERSION,
        },
        deep=True,
    )
