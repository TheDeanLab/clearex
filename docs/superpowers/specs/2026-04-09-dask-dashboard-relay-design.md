# 2026-04-09 Dask Dashboard Relay Design

## Status

Approved for implementation planning.

## Summary

Add an on-demand local tokenized reverse proxy for the Dask dashboard inside
ClearEx. The proxy is started only when the user presses `Open Dask Dashboard`,
binds to `127.0.0.1`, forwards the live dashboard for a ClearEx-managed Dask
client, and shuts down automatically when the corresponding client closes.

The feature must support both GUI entry points:

- the existing `Open Dask Dashboard` button in the main analysis dialog
- a new `Open Dask Dashboard` button in the `Running Analysis` progress dialog

If proxy setup fails, ClearEx shows a warning and does not fall back to the raw
Dask dashboard URL.

## Goals

- Provide a `d6d`-style localhost experience for Dask dashboards without
  requiring manual SSH tunneling from the user's personal machine.
- Start the dashboard relay only on demand when the user presses the button.
- Restrict access to a tokenized localhost endpoint.
- Shut the relay down automatically when the corresponding ClearEx-managed Dask
  client closes.
- Support dashboards for Dask backends that ClearEx itself creates:
  - `LocalCluster`
  - `SLURMRunner`
  - `SLURMCluster`
- Add dashboard launch access from the `Running Analysis` progress dialog.
- Use the live Dask client's real dashboard address rather than guessed config
  strings.

## Non-goals

- Do not support Dask clusters created by unrelated external programs.
- Do not keep a relay alive for the whole GUI session once the client is gone.
- Do not open the raw scheduler dashboard URL as a fallback when proxy setup
  fails.
- Do not expose the relay on non-local interfaces.
- Do not require users to run or understand external proxy infrastructure in
  the first version.

## Context

Current ClearEx behavior is config-driven and browser-only:

- the analysis dialog includes an `Open Dask Dashboard` button
- button state and URL resolution are derived from static backend config and
  scheduler-file parsing in `src/clearex/gui/app.py`
- the click handler directly calls `webbrowser.open_new_tab(...)`

This is insufficient on HPC systems where the dashboard host is not directly
reachable from the user's browser.

There is also a correctness gap in the current local path:

- `src/clearex/io/experiment.py` creates local clients with
  `dashboard_address=":0"` by default
- the GUI currently resolves `LocalCluster` to `127.0.0.1:8787`

That means the button can be wrong even before network restrictions are
considered.

The `d6d` repository already uses a session-local localhost server pattern:

- reserve a local port
- generate a random token
- serve only on `127.0.0.1`
- open a tokenized localhost URL

That pattern is appropriate for ClearEx, but ClearEx needs an actual reverse
proxy because the goal is to display the Dask dashboard itself rather than a
custom application page.

## Chosen Approach

Add a GUI-owned dashboard relay manager that starts a per-client localhost
reverse proxy on demand.

This is preferred over direct raw-URL opening because it solves the HPC
reachability problem and gives the required one-click UX. It is preferred over
depending on external Jupyter-style proxy infrastructure because the desired
workflow is self-contained inside ClearEx. It is also preferred over a global
always-on service because the user explicitly wants the relay to exist only
while the corresponding Dask client is alive.

## User Experience Contract

### Main analysis dialog

- Keep the existing `Open Dask Dashboard` button.
- The button should no longer be enabled merely because backend settings imply
  a dashboard URL.
- The button is enabled only when the dialog has access to a current
  ClearEx-managed Dask client whose dashboard can be proxied.
- Clicking the button starts the relay on first use and opens the tokenized
  localhost URL in the browser.
- Clicking again for the same still-live client reopens the existing local URL.

### Running Analysis dialog

- Add a new `Open Dask Dashboard` button beside `Stop Analysis`.
- The button is enabled only while the running workflow has a live
  ClearEx-managed Dask client registered with the relay manager.
- This gives the user a guaranteed launch point while a Dask-backed analysis is
  actively executing.

### Failure behavior

- If no eligible ClearEx-managed Dask client is active, show a warning.
- If the relay cannot be started, show a warning.
- If the Dask dashboard origin is unreachable from the ClearEx host/process,
  show a warning.
- Do not fall back to opening the raw Dask dashboard URL directly.

## Relay Architecture

### Ownership model

The relay is owned by the GUI process and managed through a shared
`DashboardRelayManager`.

Responsibilities:

- track live ClearEx-managed Dask clients
- resolve the true dashboard target for each client
- lazily create a localhost relay session when requested
- reopen an existing relay for the same client while it remains alive
- stop relay sessions when their tracked clients close
- stop all remaining relay sessions on application shutdown

### Client scope

Only ClearEx-created clients are eligible. The manager does not scan for or
attach to foreign Dask clients created by other programs.

Eligible creation paths:

- analysis/runtime startup in `src/clearex/main.py`
- GUI-side backend connection helpers in `src/clearex/gui/app.py`

The manager should receive explicit registration and unregistration events from
those code paths rather than trying to infer liveness from static workflow
config alone.

### Session model

Each eligible Dask client may have at most one active relay session.

A relay session contains:

- `session_id`
- token
- local bind host and port
- resolved upstream dashboard origin
- client identity / owner id
- created timestamp
- server handle
- shutdown callback / liveness hook

Per-client relay sessions are preferred over a single multiplexed dashboard
server because they more closely match the `d6d` session pattern, simplify
ownership, and make teardown logic unambiguous.

## Target Resolution

The relay must use the live Dask client's actual dashboard target.

Resolution order:

1. `client.dashboard_link`
2. derived scheduler metadata if the client property is missing
3. failure with warning

Static GUI config is not authoritative for launch because:

- local dashboards may bind to ephemeral ports
- scheduler files may omit the best externally reachable URL
- runtime-created clients are the real source of truth

The resolved target should be normalized to an HTTP/HTTPS origin plus path.

## Reverse Proxy Contract

### Network binding

- bind only to `127.0.0.1`
- use an ephemeral local port
- never bind on `0.0.0.0`

### Request forwarding

The proxy must forward all dashboard traffic needed for normal use:

- HTML pages
- static assets
- query strings
- redirects
- websocket traffic

This is required because the Dask dashboard uses Bokeh/Tornado endpoints that
are not satisfied by simple HTML rewriting alone.

### Upstream reachability

The proxy succeeds only if the ClearEx process can reach the dashboard origin.
If the GUI host cannot reach the scheduler dashboard, the button should warn
the user rather than opening a broken page.

### Redirect and path handling

The proxy must keep the browser on the local tokenized URL instead of allowing
redirects or absolute links to escape to the raw scheduler host.

Implementation requirements:

- preserve relative paths
- rewrite `Location` headers that point at the upstream origin
- forward websocket upgrade requests under the local origin
- preserve downstream dashboard paths such as `/status` and any deeper Bokeh
  application routes

## Authentication Contract

### Token generation

- generate a cryptographically strong random token
- match the `d6d` entropy level using `secrets.token_urlsafe(24)`

### Initial open flow

- the first browser launch uses a tokenized URL such as
  `http://127.0.0.1:<port>/?token=<token>`
- successful token validation sets an HttpOnly session cookie

### Subsequent navigation

- after the initial handshake, normal dashboard navigation should use the
  cookie rather than requiring the token on every path
- requests without a valid token or cookie are rejected

### Security posture

This is localhost-only convenience security, not an internet-facing auth
system. The important guarantees are:

- localhost-only binding
- unguessable session token
- no unauthenticated dashboard access through the proxy

## Implementation Shape

Add a focused GUI-side module:

- `src/clearex/gui/dask_dashboard_proxy.py`

Recommended responsibilities:

- `DashboardRelayManager`
- `DashboardRelaySession`
- token/cookie validation helpers
- upstream URL normalization helpers
- local relay server startup/shutdown helpers

`src/clearex/gui/app.py` should keep only thin integration code:

- register/unregister clients
- button enablement wiring
- warning dialogs
- browser open calls

## Dependency Strategy

Avoid introducing a new FastAPI/Uvicorn application stack for this feature.

Preferred implementation:

- use a lightweight Tornado-based proxy server

Reasoning:

- Dask/Bokeh already depend on Tornado concepts and websocket transport
- ClearEx already carries `bokeh` and `jupyterlab`
- a Tornado implementation should minimize dependency churn while handling HTTP
  and websocket proxying correctly

If Tornado proves insufficient in practice, a small explicit HTTP/websocket
proxy dependency can be evaluated during implementation, but that is not the
preferred first path.

## GUI Integration Details

### Main analysis dialog changes

- replace config-only dashboard button logic with active-client-aware logic
- button tooltip should describe the active local relay URL when available
- if no client is active, tooltip should explain that the dashboard is only
  available while a ClearEx-managed Dask backend is running

### Running Analysis dialog changes

- add `Open Dask Dashboard` button
- place it in the footer row with `Stop Analysis`
- keep styling consistent with existing themed buttons
- wire the button to the same relay-launch method used by the main analysis
  dialog

### Active client registry

Introduce a small GUI/runtime registry for active ClearEx-managed Dask clients.

Stored state should include:

- backend mode
- client object or weak reference
- dashboard target URL
- owner context
  - main analysis dialog
  - running-analysis dialog / workflow execution
- optional relay session handle

The registry must be updated from the same code paths that create and destroy
Dask clients so enablement state stays accurate.

## Lifecycle And Cleanup

Relay teardown triggers:

- corresponding Dask client closes
- corresponding workflow execution ends
- owner dialog is destroyed
- application shutdown

Cleanup rules:

- stop the local relay server
- clear the session token/cookie state
- release the local port
- remove the relay record from the manager

If a stale relay record is discovered during a new launch attempt, ClearEx
should discard it and create a fresh relay instead of trying to resurrect it.

## Error Handling

Warning conditions should include clear diagnostics for:

- no eligible ClearEx-managed Dask client is active
- dashboard target URL could not be resolved from the live client
- localhost relay port could not be reserved
- upstream dashboard is unreachable from the GUI host/process
- websocket forwarding could not be established
- browser launch failed after a successful relay start

Warnings should remain user-facing and concise, while detailed exceptions can be
included behind the existing themed error/warning affordances when useful.

## Documentation Updates

Update in the same implementation change set:

- `src/clearex/AGENTS.md`
- `src/clearex/gui/README.md`
- relevant runtime docs under `docs/source/runtime/`

Documentation must describe:

- that ClearEx now uses a localhost tokenized proxy for dashboard launch
- that the dashboard is available only while a ClearEx-managed Dask client is
  alive
- that the button warns instead of opening raw URLs when proxy setup fails

## Testing Plan

### Unit tests

- dashboard target normalization from live client values
- token validation and cookie issuance
- rejection of missing or invalid auth
- stale relay cleanup behavior

### Proxy integration tests

- HTTP request forwarding to a lightweight local upstream server
- websocket forwarding through the relay
- redirect rewriting back to the local origin
- rejection when the upstream origin is unreachable

### GUI tests

- main analysis dialog button enable/disable behavior
- running-analysis dialog button presence and enable/disable behavior
- warning behavior when no active client exists
- repeated clicks reuse an existing relay session for the same live client

### Runtime integration tests

- client registration on ClearEx-managed backend startup
- relay teardown on client close
- relay teardown when workflow execution ends

### Manual validation

- `LocalCluster` dashboard launch
- `SLURMRunner` dashboard launch when reachable from the GUI host
- `SLURMCluster` dashboard launch when reachable from the GUI host
- running-analysis dialog launch during an active Dask-backed workflow

## Risks And Mitigations

### Websocket proxy complexity

Risk:
- partial proxy implementations often work for static pages but fail for live
  dashboard components

Mitigation:
- treat websocket forwarding as a first-class requirement
- add targeted integration tests before claiming support

### Stale client or relay state

Risk:
- GUI state may say a dashboard is available after the client is already gone

Mitigation:
- explicit registration/unregistration hooks
- stale-session detection and forced cleanup on launch attempts

### Dependency creep

Risk:
- solving this with a full new web stack would expand maintenance burden

Mitigation:
- prefer Tornado-based implementation first
- justify any new dependency explicitly if later required

## Final Decision

Implement an on-demand, localhost-only, tokenized reverse proxy for Dask
dashboards in ClearEx that:

- serves only dashboards for Dask clients that ClearEx itself launches
- starts only when the user presses `Open Dask Dashboard`
- supports both the main analysis dialog and the running-analysis dialog
- proxies HTTP and websocket dashboard traffic
- shuts down automatically when the corresponding Dask client closes
- warns on failure instead of opening raw dashboard URLs
