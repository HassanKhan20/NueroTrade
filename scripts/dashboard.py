"""Launch the local dashboard.

  python -m scripts.dashboard
  python -m scripts.dashboard --port 8765 --host 0.0.0.0

Open http://127.0.0.1:8000/ in a browser. Refreshes every 2 seconds.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true",
                   help="auto-reload on file changes (dev only)")
    args = p.parse_args(argv)
    import uvicorn
    uvicorn.run(
        "src.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
