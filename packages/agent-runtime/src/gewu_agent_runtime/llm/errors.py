"""Provider-neutral failures raised by language-model adapters."""


class ModelInvocationError(Exception):
    """Base failure for one model service invocation."""


class ModelAuthenticationError(ModelInvocationError):
    """The configured model credentials were rejected."""


class ModelPermissionDeniedError(ModelInvocationError):
    """The configured credentials cannot access the requested model."""


class ModelRateLimitError(ModelInvocationError):
    """The model service rejected an invocation because of rate limits."""


class ModelTimeoutError(ModelInvocationError):
    """The model service did not finish within the configured timeout."""


class ModelUnavailableError(ModelInvocationError):
    """The model service could not be reached or failed transiently."""


class ModelRequestRejectedError(ModelInvocationError):
    """The model service rejected a request for a non-retryable reason."""


def model_error_for_status(status_code: int) -> ModelInvocationError:
    """Return a safe provider-neutral failure for one HTTP status code."""

    if status_code == 401:
        return ModelAuthenticationError()
    if status_code == 403:
        return ModelPermissionDeniedError()
    if status_code == 408:
        return ModelTimeoutError()
    if status_code == 429:
        return ModelRateLimitError()
    if status_code >= 500:
        return ModelUnavailableError()
    return ModelRequestRejectedError()
