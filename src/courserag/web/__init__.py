"""The local web UI.

Kept in its own subpackage and imported only by ``kb web``, so the ``[web]`` extra's
dependencies (fastapi, uvicorn) stay off every other code path. Importing *this*
module is still free — ``create_app`` and ``serve`` are resolved lazily on attribute
access — so a test can check the package exists without needing the extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from courserag.web.app import create_app, serve

__all__ = ["create_app", "serve"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from courserag.web import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
