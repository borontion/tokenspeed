# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

_ACTIVE_KERNEL_CONTEXT: ContextVar[KernelContext | None] = ContextVar(
    "tokenspeed_kernel_context",
    default=None,
)


@dataclass
class KernelContext:
    """Scoped, namespaced kernel-owned state.

    Operator APIs stay stateless. Runtime backends create a context for the
    lifetime they own, enter it around kernel calls, and runtime/backend-specific
    code stores state under unique namespaces.
    """

    _namespaces: dict[str, dict[str, Any]] = field(default_factory=dict)

    @contextmanager
    def use(self) -> Iterator[KernelContext]:
        token = _ACTIVE_KERNEL_CONTEXT.set(self)
        try:
            yield self
        finally:
            _ACTIVE_KERNEL_CONTEXT.reset(token)

    def namespace(self, name: str) -> dict[str, Any]:
        if not name:
            raise ValueError("KernelContext namespace name must be non-empty")
        return self._namespaces.setdefault(name, {})

    def reset(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._namespaces.clear()
        else:
            self._namespaces.pop(namespace, None)


def _current_kernel_context() -> KernelContext | None:
    return _ACTIVE_KERNEL_CONTEXT.get()


__all__ = ["KernelContext", "_current_kernel_context"]
