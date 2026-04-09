from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

from tornado import httpserver, netutil, web, websocket
from tornado.ioloop import IOLoop
from tornado.websocket import websocket_connect

import pytest

from clearex.gui.dask_dashboard_proxy import (
    DashboardRelayManager,
    resolve_client_dashboard_url,
)


def test_resolve_client_dashboard_url_prefers_live_dashboard_link() -> None:
    client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status"
    )

    resolved = resolve_client_dashboard_url(client)

    assert resolved == "http://scheduler.example.internal:8787/status"


def test_resolve_client_dashboard_url_rejects_missing_dashboard_link() -> None:
    client = SimpleNamespace(dashboard_link=None)

    with pytest.raises(ValueError, match="dashboard"):
        resolve_client_dashboard_url(client)


def test_dashboard_relay_manager_reuses_existing_session(monkeypatch) -> None:
    manager = DashboardRelayManager()
    client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status",
        status="running",
    )
    started: list[tuple[str, str]] = []

    def _fake_start_session(*, client_id: str, upstream_url: str):
        started.append((client_id, upstream_url))
        return SimpleNamespace(
            client_id=client_id,
            token="token-123",
            local_url="http://127.0.0.1:40000/?token=token-123",
            upstream_url=upstream_url,
            close=lambda: None,
        )

    monkeypatch.setattr(manager, "_start_session", _fake_start_session)

    manager.register_client(
        workload="analysis", backend_mode="local_cluster", client=client
    )
    first = manager.open_dashboard(workload="analysis")
    second = manager.open_dashboard(workload="analysis")

    assert first == "http://127.0.0.1:40000/?token=token-123"
    assert second == first
    assert len(started) == 1


def test_dashboard_relay_manager_unregister_closes_session(monkeypatch) -> None:
    manager = DashboardRelayManager()
    client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status",
        status="running",
    )
    closed: list[str] = []

    def _fake_start_session(*, client_id: str, upstream_url: str):
        del upstream_url
        return SimpleNamespace(
            client_id=client_id,
            token="token-123",
            local_url="http://127.0.0.1:41000/?token=token-123",
            upstream_url="http://scheduler.example.internal:8787/status",
            close=lambda: closed.append(client_id),
        )

    monkeypatch.setattr(manager, "_start_session", _fake_start_session)

    client_id = manager.register_client(
        workload="analysis",
        backend_mode="slurm_cluster",
        client=client,
    )
    _ = manager.open_dashboard(workload="analysis")

    manager.unregister_client(client_id)

    assert closed == [client_id]
    assert manager.has_available_client(workload="analysis") is False


def test_dashboard_relay_manager_rejects_stopped_client_and_purges_registration(
    monkeypatch,
) -> None:
    manager = DashboardRelayManager()
    client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status",
        status="running",
    )

    def _fake_start_session(*, client_id: str, upstream_url: str):
        del upstream_url
        return SimpleNamespace(
            client_id=client_id,
            token="token-123",
            local_url="http://127.0.0.1:42000/?token=token-123",
            upstream_url="http://scheduler.example.internal:8787/status",
            close=lambda: None,
        )

    monkeypatch.setattr(manager, "_start_session", _fake_start_session)

    client_id = manager.register_client(
        workload="analysis",
        backend_mode="local_cluster",
        client=client,
    )
    client.status = "closed"

    assert manager.has_available_client(workload="analysis") is False
    assert client_id not in manager._clients
    assert client_id not in manager._sessions

    with pytest.raises(ValueError, match="No active ClearEx-managed Dask client"):
        manager.open_dashboard(workload="analysis")


def test_dashboard_relay_manager_unregister_clears_state_when_close_fails(
    monkeypatch,
) -> None:
    manager = DashboardRelayManager()
    client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status",
        status="running",
    )

    def _fake_start_session(*, client_id: str, upstream_url: str):
        del upstream_url

        def _raise_close() -> None:
            raise RuntimeError(f"failed to close {client_id}")

        return SimpleNamespace(
            client_id=client_id,
            token="token-123",
            local_url="http://127.0.0.1:43000/?token=token-123",
            upstream_url="http://scheduler.example.internal:8787/status",
            close=_raise_close,
        )

    monkeypatch.setattr(manager, "_start_session", _fake_start_session)

    client_id = manager.register_client(
        workload="analysis",
        backend_mode="local_cluster",
        client=client,
    )
    _ = manager.open_dashboard(workload="analysis")

    with pytest.raises(RuntimeError, match=f"failed to close {client_id}"):
        manager.unregister_client(client_id)

    assert client_id not in manager._clients
    assert client_id not in manager._sessions


def test_dashboard_relay_manager_shutdown_continues_after_close_failure(
    monkeypatch,
) -> None:
    manager = DashboardRelayManager()
    closed: list[str] = []

    def _fake_start_session(*, client_id: str, upstream_url: str):
        del upstream_url

        def _close() -> None:
            if client_id == first_client_id:
                raise RuntimeError(f"failed to close {client_id}")
            closed.append(client_id)

        return SimpleNamespace(
            client_id=client_id,
            token=f"token-{client_id}",
            local_url=f"http://127.0.0.1:44{len(closed)}00/?token=token-{client_id}",
            upstream_url="http://scheduler.example.internal:8787/status",
            close=_close,
        )

    monkeypatch.setattr(manager, "_start_session", _fake_start_session)

    first_client_id = manager.register_client(
        workload="analysis",
        backend_mode="local_cluster",
        client=SimpleNamespace(
            dashboard_link="http://scheduler.example.internal:8787/status",
            status="running",
        ),
    )
    second_client_id = manager.register_client(
        workload="segmentation",
        backend_mode="local_cluster",
        client=SimpleNamespace(
            dashboard_link="http://scheduler.example.internal:8787/status",
            status="running",
        ),
    )
    _ = manager.open_dashboard(workload="analysis")
    _ = manager.open_dashboard(workload="segmentation")

    manager.shutdown()

    assert closed == [second_client_id]
    assert not manager._clients
    assert not manager._sessions


def test_dashboard_relay_manager_skips_stale_client_when_purge_close_fails(
    monkeypatch,
) -> None:
    manager = DashboardRelayManager()
    healthy_client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status",
        status="running",
    )
    stale_client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status",
        status="closed",
    )
    started: list[str] = []
    closed: list[str] = []

    def _fake_start_session(*, client_id: str, upstream_url: str):
        started.append(client_id)
        del upstream_url
        if client_id == stale_client_id:

            def _raise_close() -> None:
                raise RuntimeError(f"failed to close {client_id}")

            close = _raise_close
            local_url = "http://127.0.0.1:44000/?token=token-stale"
        else:

            def _close() -> None:
                closed.append(client_id)

            close = _close
            local_url = "http://127.0.0.1:44100/?token=token-healthy"
        return SimpleNamespace(
            client_id=client_id,
            token=f"token-{client_id}",
            local_url=local_url,
            upstream_url="http://scheduler.example.internal:8787/status",
            close=close,
        )

    monkeypatch.setattr(manager, "_start_session", _fake_start_session)

    healthy_client_id = manager.register_client(
        workload="analysis",
        backend_mode="local_cluster",
        client=healthy_client,
    )
    stale_client_id = manager.register_client(
        workload="analysis",
        backend_mode="local_cluster",
        client=stale_client,
    )
    manager._sessions[stale_client_id] = SimpleNamespace(
        client_id=stale_client_id,
        token="token-stale",
        local_url="http://127.0.0.1:44000/?token=token-stale",
        upstream_url="http://scheduler.example.internal:8787/status",
        close=lambda: _raise_close(stale_client_id),
    )

    resolved = manager.open_dashboard(workload="analysis")

    assert resolved == "http://127.0.0.1:44100/?token=token-healthy"
    assert healthy_client_id in manager._clients
    assert stale_client_id not in manager._clients
    assert stale_client_id not in manager._sessions
    assert started == [healthy_client_id]
    assert closed == []


def _raise_close(client_id: str) -> None:
    raise RuntimeError(f"failed to close {client_id}")


def test_dashboard_relay_manager_skips_client_with_invalid_refreshed_dashboard_link(
    monkeypatch,
) -> None:
    manager = DashboardRelayManager()
    healthy_client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status",
        status="running",
    )
    stale_client = SimpleNamespace(
        dashboard_link="http://scheduler.example.internal:8787/status",
        status="running",
    )
    started: list[str] = []

    def _fake_start_session(*, client_id: str, upstream_url: str):
        started.append(client_id)
        del upstream_url
        return SimpleNamespace(
            client_id=client_id,
            token=f"token-{client_id}",
            local_url="http://127.0.0.1:45000/?token=token-healthy",
            upstream_url="http://scheduler.example.internal:8787/status",
            close=lambda: None,
        )

    monkeypatch.setattr(manager, "_start_session", _fake_start_session)

    healthy_client_id = manager.register_client(
        workload="analysis",
        backend_mode="local_cluster",
        client=healthy_client,
    )
    stale_client_id = manager.register_client(
        workload="analysis",
        backend_mode="local_cluster",
        client=stale_client,
    )
    stale_client.dashboard_link = ""

    assert manager.has_available_client(workload="analysis") is True
    resolved = manager.open_dashboard(workload="analysis")

    assert resolved == "http://127.0.0.1:45000/?token=token-healthy"
    assert healthy_client_id in manager._clients
    assert stale_client_id not in manager._clients
    assert stale_client_id not in manager._sessions
    assert started == [healthy_client_id]


class _UpstreamStatusHandler(web.RequestHandler):
    def get(self) -> None:
        self.write("upstream-status-ok")


class _UpstreamRedirectHandler(web.RequestHandler):
    def get(self) -> None:
        self.redirect("/status")


class _UpstreamWebSocketHandler(websocket.WebSocketHandler):
    def open(self) -> None:
        return None

    def on_message(self, message: str) -> None:
        self.write_message(f"echo:{message}")


def _run_tornado_app(
    application: web.Application,
) -> tuple[IOLoop, int, threading.Thread]:
    asyncio_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(asyncio_loop)
    loop = IOLoop(asyncio_loop=asyncio_loop)
    asyncio.set_event_loop(None)
    sockets = netutil.bind_sockets(0, "127.0.0.1")
    port = sockets[0].getsockname()[1]
    server = httpserver.HTTPServer(application)
    ready = threading.Event()

    def _start() -> None:
        asyncio.set_event_loop(asyncio_loop)
        server.add_sockets(sockets)
        ready.set()
        loop.start()

    thread = threading.Thread(target=_start, daemon=True)
    thread.start()
    assert ready.wait(timeout=5), "upstream tornado app did not start"
    return loop, port, thread


def _stop_tornado_app(loop: IOLoop, thread: threading.Thread) -> None:
    loop.add_callback(loop.stop)
    thread.join(timeout=5)


def test_dashboard_proxy_requires_token() -> None:
    upstream_loop, upstream_port, upstream_thread = _run_tornado_app(
        web.Application([(r"/status", _UpstreamStatusHandler)])
    )
    manager = DashboardRelayManager()
    client = SimpleNamespace(
        dashboard_link=f"http://127.0.0.1:{upstream_port}/status",
        status="running",
    )
    manager.register_client(
        workload="analysis", backend_mode="local_cluster", client=client
    )
    tokenized_url = manager.open_dashboard(workload="analysis")
    unauthorized = tokenized_url.split("?", 1)[0]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            opener.open(unauthorized)

        assert excinfo.value.code in {401, 403}
    finally:
        manager.shutdown()
        _stop_tornado_app(upstream_loop, upstream_thread)


def test_dashboard_proxy_forwards_http_and_rewrites_redirects() -> None:
    upstream_loop, upstream_port, upstream_thread = _run_tornado_app(
        web.Application(
            [
                (r"/status", _UpstreamStatusHandler),
                (r"/redirect", _UpstreamRedirectHandler),
            ]
        )
    )
    manager = DashboardRelayManager()
    client = SimpleNamespace(
        dashboard_link=f"http://127.0.0.1:{upstream_port}/redirect",
        status="running",
    )
    manager.register_client(
        workload="analysis", backend_mode="local_cluster", client=client
    )
    tokenized_url = manager.open_dashboard(workload="analysis")

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(),
    )

    try:
        response = opener.open(tokenized_url)
        payload = response.read().decode("utf-8")

        assert "upstream-status-ok" in payload
        assert f"127.0.0.1:{upstream_port}" not in response.geturl()

        follow_up = opener.open(tokenized_url.rsplit("/", 1)[0] + "/status")
        assert "upstream-status-ok" in follow_up.read().decode("utf-8")
    finally:
        manager.shutdown()
        _stop_tornado_app(upstream_loop, upstream_thread)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="websocket test uses POSIX event loop assumptions",
)
def test_dashboard_proxy_forwards_websocket_messages() -> None:
    upstream_loop, upstream_port, upstream_thread = _run_tornado_app(
        web.Application([(r"/ws", _UpstreamWebSocketHandler)])
    )
    manager = DashboardRelayManager()
    client = SimpleNamespace(
        dashboard_link=f"http://127.0.0.1:{upstream_port}/status",
        status="running",
    )
    manager.register_client(
        workload="analysis", backend_mode="local_cluster", client=client
    )
    tokenized_url = manager.open_dashboard(workload="analysis")
    base_url, query = tokenized_url.split("?", 1)
    ws_url = base_url.rsplit("/", 1)[0].replace("http://", "ws://") + f"/ws?{query}"

    async def _exercise() -> str:
        conn = await websocket_connect(ws_url)
        await conn.write_message("hello")
        message = await conn.read_message()
        conn.close()
        return str(message)

    try:
        echoed = asyncio.run(_exercise())

        assert echoed == "echo:hello"
    finally:
        manager.shutdown()
        _stop_tornado_app(upstream_loop, upstream_thread)
