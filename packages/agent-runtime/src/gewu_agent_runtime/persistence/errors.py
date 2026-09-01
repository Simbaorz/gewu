"""Stable persistence failures surfaced by runtime stores."""


class PersistenceError(Exception):
    """Base persistence error."""


class EntityNotFoundError(PersistenceError):
    """Requested runtime entity does not exist."""


class ConcurrentWriteError(PersistenceError):
    """A compare-and-set precondition was no longer current."""


class MessageWriteConflictError(ConcurrentWriteError):
    """One or more append-only message identities already exist."""


class IdempotencyConflictError(PersistenceError):
    """An idempotency key was reused for a different request."""
