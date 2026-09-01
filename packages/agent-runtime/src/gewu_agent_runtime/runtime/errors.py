"""Stable failures from the high-level Runtime API."""


class RuntimeErrorBase(Exception):
    """Base high-level Runtime failure."""


class SafeExecutionError(RuntimeErrorBase):
    """Known failure whose audited code and message may be returned to callers."""

    def __init__(self, *, code: str, message: str) -> None:
        normalized_code = code.strip()
        normalized_message = message.strip()
        if not normalized_code or len(normalized_code) > 64:
            raise ValueError("Safe execution error code must contain 1 to 64 characters.")
        if not normalized_message:
            raise ValueError("Safe execution error message cannot be empty.")
        super().__init__(normalized_message)
        self.code = normalized_code
        self.message = normalized_message


class SubscriberMismatchError(RuntimeErrorBase):
    """An invocation crossed subscriber isolation."""


class PrincipalMismatchError(RuntimeErrorBase):
    """An invocation crossed principal isolation without explicit delegation."""


class ConcurrentRunError(RuntimeErrorBase):
    """Another run currently owns the conversation."""


class AskNotPendingError(RuntimeErrorBase):
    """The requested Ask continuation is no longer pending."""


class AskExpiredError(RuntimeErrorBase):
    """The requested Ask continuation has expired."""
