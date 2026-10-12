from __future__ import annotations

import json
import logging
from pathlib import Path

from src.utils.logging import configure_logging, get_logger


def test_log_to_file(tmp_path: Path) -> None:
    logfile = tmp_path / "app.log"
    configure_logging("INFO", logfile=logfile)
    log = get_logger("test")
    log.info("hello", extra={"k": 1})
    # tear down so the handler releases the file on Windows
    logging.getLogger().handlers.clear()
    text = logfile.read_text(encoding="utf-8").strip()
    line = json.loads(text.splitlines()[-1])
    assert line["msg"] == "hello"
    assert line["level"] == "INFO"
    assert line["k"] == 1
