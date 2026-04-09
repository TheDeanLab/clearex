from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import secrets
import threading
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse
from uuid import uuid4

from tornado import httpclient, httpserver, ioloop, netutil, web, websocket


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


def _request_token_from_query(query: str) -> str:
    filtered = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key != "token"
    ]
    if not filtered:
        return ""
    return urlencode(filtered, doseq=True)


def _localize_redirect(
    *, redirect_url: str, local_origin: str, upstream_origin: str, request_token: str
) -> str:
    parsed_redirect = urlparse(redirect_url)
    parsed_local_origin = urlparse(local_origin)
    parsed_upstream_origin = urlparse(upstream_origin)
    if parsed_redirect.scheme and parsed_redirect.netloc:
        if (
            parsed_redirect.scheme == parsed_upstream_origin.scheme
            and parsed_redirect.netloc == parsed_upstream_origin.netloc
        ):
            rewritten = parsed_redirect._replace(
                scheme=parsed_local_origin.scheme,
                netloc=parsed_local_origin.netloc,
            )
        else:
            rewritten = parsed_redirect
    else:
        rewritten = urlparse(urljoin(local_origin.rstrip("/") + "/", redirect_url))

    query = rewritten.query
    if request_token and "token=" not in query:
        query = f"{query}&token={request_token}" if query else f"token={request_token}"
    return rewritten._replace(query=query).geturl()


class _ProxyBaseHandler(web.RequestHandler):
    def initialize(
        self, *, token: str, upstream_origin: str, upstream_path: str
    ) -> None:
        self._token = token
        self._upstream_origin = upstream_origin.rstrip("/")
        self._upstream_path = upstream_path or "/status"
        self._request_token = ""

    def _authorized(self) -> bool:
        query_token = str(self.get_argument("token", default="") or "").strip()
        cookie_token = str(
            self.get_cookie("clearex_dashboard_token", default="") or ""
        ).strip()
        if query_token == self._token:
            self._request_token = query_token
            self.set_cookie(
                "clearex_dashboard_token",
                self._token,
                httponly=True,
                samesite="Strict",
                path="/",
            )
            return True
        if cookie_token and cookie_token == self._token:
            return True
        return False

    def prepare(self) -> None:
        if self._authorized():
            return
        self.set_status(403)
        self.finish("Dashboard relay token is required.")


class _ProxyRequestHandler(_ProxyBaseHandler):
    async def get(self, proxied_path: str = "") -> None:
        del proxied_path
        await self._proxy_request()

    async def post(self, proxied_path: str = "") -> None:
        del proxied_path
        await self._proxy_request()

    async def put(self, proxied_path: str = "") -> None:
        del proxied_path
        await self._proxy_request()

    async def patch(self, proxied_path: str = "") -> None:
        del proxied_path
        await self._proxy_request()

    async def delete(self, proxied_path: str = "") -> None:
        del proxied_path
        await self._proxy_request()

    async def head(self, proxied_path: str = "") -> None:
        del proxied_path
        await self._proxy_request()

    async def _proxy_request(self) -> None:
        upstream_path = self.request.path or "/"
        if upstream_path == "/":
            upstream_path = self._upstream_path

        request_query = _request_token_from_query(self.request.query)
        upstream_url = f"{self._upstream_origin}{upstream_path}"
        if request_query:
            upstream_url = f"{upstream_url}?{request_query}"

        headers = {}
        for name, value in self.request.headers.get_all():
            header_name = name.lower()
            if header_name in {
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
                "cookie",
            }:
                continue
            headers[name] = value

        response = await httpclient.AsyncHTTPClient().fetch(
            httpclient.HTTPRequest(
                url=upstream_url,
                method=self.request.method,
                headers=headers,
                body=self.request.body if self.request.body else None,
                follow_redirects=False,
                request_timeout=30.0,
            ),
            raise_error=False,
        )

        self.set_status(response.code)
        for header, value in response.headers.get_all():
            header_name = header.lower()
            if header_name == "location":
                value = _localize_redirect(
                    redirect_url=str(value),
                    local_origin=self.request.full_url().split("?", 1)[0],
                    upstream_origin=self._upstream_origin,
                    request_token=self._request_token,
                )
            if header_name in {"content-length", "transfer-encoding", "connection"}:
                continue
            self.set_header(header, value)

        if self.request.method != "HEAD" and response.body:
            self.finish(response.body)
        else:
            self.finish()


class _ProxyWebSocketHandler(_ProxyBaseHandler, websocket.WebSocketHandler):
    async def open(self, proxied_path: str = "") -> None:
        del proxied_path
        request_path = self.request.path or "/"
        request_query = _request_token_from_query(self.request.query)
        websocket_origin = (
            "wss" if urlparse(self._upstream_origin).scheme == "https" else "ws"
        )
        upstream_url = f"{websocket_origin}://{urlparse(self._upstream_origin).netloc}{request_path}"
        if request_query:
            upstream_url = f"{upstream_url}?{request_query}"

        self._upstream = await websocket.websocket_connect(upstream_url)
        self._upstream_task = asyncio.create_task(self._pump_upstream())

    async def _pump_upstream(self) -> None:
        try:
            while True:
                message = await self._upstream.read_message()
                if message is None:
                    break
                await self.write_message(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def on_message(self, message: str) -> None:
        await self._upstream.write_message(message)

    def on_close(self) -> None:
        upstream = getattr(self, "_upstream", None)
        if upstream is not None:
            upstream.close()
        task = getattr(self, "_upstream_task", None)
        if task is not None and not task.done():
            task.cancel()


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
        token = secrets.token_urlsafe(24)
        parsed_upstream = urlparse(upstream_url)
        upstream_origin = f"{parsed_upstream.scheme}://{parsed_upstream.netloc}"
        upstream_path = parsed_upstream.path or "/status"

        sockets = netutil.bind_sockets(0, "127.0.0.1")
        port = sockets[0].getsockname()[1]
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)
        loop = ioloop.IOLoop(asyncio_loop=asyncio_loop)
        asyncio.set_event_loop(None)
        application = web.Application(
            [
                (
                    r"/(?:.+/)?ws(?:/.*)?",
                    _ProxyWebSocketHandler,
                    {
                        "token": token,
                        "upstream_origin": upstream_origin,
                        "upstream_path": upstream_path,
                    },
                ),
                (
                    r"/(.*)",
                    _ProxyRequestHandler,
                    {
                        "token": token,
                        "upstream_origin": upstream_origin,
                        "upstream_path": upstream_path,
                    },
                ),
            ]
        )
        server = httpserver.HTTPServer(application)
        started = threading.Event()
        stopped = threading.Event()

        def _run_loop() -> None:
            try:
                asyncio.set_event_loop(asyncio_loop)
                server.add_sockets(sockets)
                started.set()
                loop.start()
            finally:
                for socket in sockets:
                    try:
                        socket.close()
                    except Exception:
                        pass
                stopped.set()

        thread = threading.Thread(
            target=_run_loop,
            daemon=True,
            name=f"dashboard-relay-{client_id[:8]}",
        )
        thread.start()

        if not started.wait(timeout=5):
            raise RuntimeError("Relay session startup timed out.")

        def _close() -> None:
            if stopped.is_set():
                return

            def _shutdown() -> None:
                server.stop()
                loop.stop()

            try:
                loop.add_callback(_shutdown)
            except RuntimeError:
                pass
            if thread.is_alive() and threading.current_thread() is not thread:
                thread.join(timeout=5)

        local_url = f"http://127.0.0.1:{port}{upstream_path}?token={token}"
        return DashboardRelaySession(
            client_id=client_id,
            token=token,
            local_url=local_url,
            upstream_url=upstream_url,
            close=_close,
        )
