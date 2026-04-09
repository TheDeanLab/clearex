from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import uuid4


def resolve_client_dashboard_url(client: Any) -> str:
    raw = str(getattr(client, "dashboard_link", "") or "").strip()
    if not raw:
        raise ValueError("Live Dask client does not expose a dashboard link.")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid dashboard link: {raw!r}")
    return raw


@dataclass
class RegisteredDashboardClient:
    client_id: str
    workload: str
    backend_mode: str
    client: Any
    upstream_url: str


@dataclass
class DashboardRelaySession:
    client_id: str
    token: str
    local_url: str
    upstream_url: str
    close: Any


class DashboardRelayManager:
    def __init__(self) -> None:
        self._clients: "OrderedDict[str, RegisteredDashboardClient]" = OrderedDict()
        self._sessions: dict[str, DashboardRelaySession] = {}

    def register_client(self, *, workload: str, backend_mode: str, client: Any) -> str:
        client_id = uuid4().hex
        self._clients[client_id] = RegisteredDashboardClient(
            client_id=client_id,
            workload=str(workload).strip().lower(),
            backend_mode=str(backend_mode).strip().lower(),
            client=client,
            upstream_url=resolve_client_dashboard_url(client),
        )
        return client_id

    def unregister_client(self, client_id: str) -> None:
        client_key = str(client_id)
        session = self._sessions.pop(client_key, None)
        try:
            if session is not None:
                session.close()
        finally:
            self._clients.pop(client_key, None)

    def has_available_client(self, *, workload: Optional[str] = None) -> bool:
        return self._select_client_id(workload=workload) is not None

    def open_dashboard(self, *, workload: Optional[str] = None) -> str:
        client_id = self._select_client_id(workload=workload)
        if client_id is None:
            raise ValueError("No active ClearEx-managed Dask client is available.")
        session = self._sessions.get(client_id)
        if session is None:
            registered = self._clients[client_id]
            session = self._start_session(
                client_id=client_id,
                upstream_url=registered.upstream_url,
            )
            self._sessions[client_id] = session
        return str(session.local_url)

    def shutdown(self) -> None:
        for client_id in list(self._clients):
            try:
                self.unregister_client(client_id)
            except Exception:
                continue

    def _select_client_id(self, *, workload: Optional[str]) -> Optional[str]:
        requested = str(workload).strip().lower() if workload is not None else None
        for client_id in list(self._clients.keys())[::-1]:
            registered = self._clients.get(client_id)
            if registered is None:
                continue
            if requested is not None and registered.workload != requested:
                continue
            refreshed = False
            try:
                refreshed = self._refresh_registered_client(registered)
            except Exception:
                refreshed = False
            finally:
                if not refreshed:
                    try:
                        self.unregister_client(client_id)
                    except Exception:
                        pass
            if refreshed:
                return client_id
        return None

    def _refresh_registered_client(self, registered: RegisteredDashboardClient) -> bool:
        client = registered.client
        status = str(getattr(client, "status", "") or "").strip().lower()
        if status != "running":
            return False
        registered.upstream_url = resolve_client_dashboard_url(client)
        return True

    def _start_session(
        self, *, client_id: str, upstream_url: str
    ) -> DashboardRelaySession:
        raise NotImplementedError("Relay session startup is not available yet.")
