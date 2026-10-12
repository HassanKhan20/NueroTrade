"""FastAPI dashboard.

Routes:
  GET /             -> live progress dashboard (auto-refresh via JS)
  GET /api/progress -> full snapshot JSON
  GET /api/state    -> raw bot_state.json (legacy)
  GET /api/health   -> {"ok": true}
  GET /classic      -> the original minimal bot_state HTML page
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from src.dashboard.progress import snapshot_dict
from src.live.state import read_state

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "bot_state.json"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="NueroTrade dashboard")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/progress")
def progress(expected: int | None = Query(default=None, ge=1)) -> JSONResponse:
    """Full progress snapshot. Pass ?expected=N to anchor the % bar."""
    return JSONResponse(snapshot_dict(expected))


@app.get("/api/state")
def state_json() -> JSONResponse:
    s = read_state(STATE_PATH)
    if s is None:
        return JSONResponse({"error": "no state file"}, status_code=404)
    return JSONResponse(s.__dict__)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    template = TEMPLATE_DIR / "progress.html"
    if not template.exists():
        return "<h1>NueroTrade</h1><p>template missing</p>"
    return template.read_text(encoding="utf-8")


@app.get("/classic", response_class=HTMLResponse)
def classic() -> str:
    """Minimal bot_state-only view, kept for backwards-compat."""
    s = read_state(STATE_PATH)
    if s is None:
        return _render_no_state()
    return _render_classic(s)


def _render_no_state() -> str:
    return """<!doctype html><html><head><meta http-equiv="refresh" content="2">
<title>NueroTrade</title>
<style>body{font-family:monospace;background:#000;color:#0f0;padding:20px}</style>
</head><body><h1>NueroTrade</h1><p>Waiting for bot_state.json...</p></body></html>"""


def _render_classic(s) -> str:
    pos_rows = "".join(
        f"<tr><td>{p['symbol']}</td><td>{p['qty']}</td><td>{p['avg_entry_price']:.2f}</td></tr>"
        for p in s.open_positions
    ) or "<tr><td colspan=3>(none)</td></tr>"
    attr = s.component_attribution
    return f"""<!doctype html><html><head><meta http-equiv="refresh" content="2">
<title>NueroTrade — classic</title>
<style>
body{{font-family:monospace;background:#000;color:#0f0;padding:20px}}
table{{border-collapse:collapse}}
td,th{{border:1px solid #0f0;padding:4px 8px}}
.kill{{color:#f33}}
</style>
</head><body>
<h1>NueroTrade — classic view</h1>
<p>updated_at: {s.updated_at}</p>
<p>enabled: {s.enabled} | kill_switch: <span class="{ 'kill' if s.kill_switch_state != 'ok' else '' }">{s.kill_switch_state}</span></p>
<p>equity: ${s.equity:,.2f} | cash: ${s.cash:,.2f}</p>
<p>regime_multiplier: {s.regime_multiplier:.2f}</p>
<p>today_trades: {s.today_trades} | today_pnl: ${s.today_pnl:,.2f}</p>
<h2>Open positions</h2>
<table><thead><tr><th>symbol</th><th>qty</th><th>avg_entry</th></tr></thead>
<tbody>{pos_rows}</tbody></table>
<h2>Attribution</h2>
<p>orb: ${attr.get('orb',0):,.2f} | intraday_momentum: ${attr.get('intraday_momentum',0):,.2f}</p>
<p>last_event: {s.last_event}</p>
</body></html>"""
