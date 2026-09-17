"""Global safety policy for the TRACE test suite.

Tests must never depend on a real OpenAI credential or external network
access. This file is loaded by pytest before test-module collection.
"""

from __future__ import annotations

import os
import socket
from typing import Any


TEST_API_KEY = "trace-v2-test-sentinel-not-a-real-key"

os.environ["OPENAI_API_KEY"] = TEST_API_KEY


class TraceTestNetworkBlocked(RuntimeError):
    """Raised when a test attempts external network access."""


_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_SENDTO = socket.socket.sendto


def _is_ip_socket(sock: socket.socket) -> bool:
    return sock.family in {
        socket.AF_INET,
        socket.AF_INET6,
    }


def _blocked(operation: str, target: Any = None) -> None:
    suffix = (
        f" target={target!r}"
        if target is not None
        else ""
    )
    raise TraceTestNetworkBlocked(
        "TRACE TEST NETWORK BLOCK: "
        f"{operation} attempted"
        f"{suffix}"
    )


def _connect(
    sock: socket.socket,
    address: Any,
) -> Any:
    if _is_ip_socket(sock):
        _blocked(
            "socket.connect",
            address,
        )

    return _ORIGINAL_CONNECT(
        sock,
        address,
    )


def _connect_ex(
    sock: socket.socket,
    address: Any,
) -> Any:
    if _is_ip_socket(sock):
        _blocked(
            "socket.connect_ex",
            address,
        )

    return _ORIGINAL_CONNECT_EX(
        sock,
        address,
    )


def _create_connection(
    address: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    _blocked(
        "socket.create_connection",
        address,
    )


def _sendto(
    sock: socket.socket,
    data: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _is_ip_socket(sock):
        target = (
            args[-1]
            if args
            else None
        )
        _blocked(
            "socket.sendto",
            target,
        )

    return _ORIGINAL_SENDTO(
        sock,
        data,
        *args,
        **kwargs,
    )


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.create_connection = _create_connection
socket.socket.sendto = _sendto
