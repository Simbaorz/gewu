"""Dependencies shared by HTTP route groups."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from gewu_core.http.password_transport import RsaPasswordTransport


def get_password_transport(request: Request) -> RsaPasswordTransport:
    """Return the startup-loaded browser password transport."""
    runtime = getattr(request.app.state, "runtime", None)
    transport = getattr(runtime, "password_transport", None)
    if not isinstance(transport, RsaPasswordTransport):
        raise RuntimeError("Password transport is not initialized.")
    return transport


PasswordTransportDep = Annotated[RsaPasswordTransport, Depends(get_password_transport)]
