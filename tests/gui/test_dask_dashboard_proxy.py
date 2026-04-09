from __future__ import annotations

from types import SimpleNamespace

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
