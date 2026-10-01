"""What the TUI shows, one timestamped line each, in data/activity.log (mirrors: data/kev/ or data/laya/).

Lab log lines, new bests, level changes, one summary per scored batch, the publisher, and snapshots, so a run can
be followed from outside the terminal (`tail -f data/activity.log`). The file rolls over at MAX_BYTES and keeps
BACKUPS older files, so it never takes more than (BACKUPS + 1) * MAX_BYTES. Every process (TUI, publisher, search)
appends to it; lines carry the process id.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from .config import DATA, EDITION

PATH = DATA / "activity.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 2

_logger: logging.Logger | None = None


def _get() -> logging.Logger | None:
    global _logger
    if _logger is None:
        logger = logging.getLogger("jevlab.activity")
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            PATH.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(PATH, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
        except OSError:
            return None
        handler.setFormatter(logging.Formatter(f"%(asctime)s {EDITION} {os.getpid()} %(source)s %(message)s",
                                               "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        _logger = logger
    return _logger


def log(source: str, message: str) -> None:
    """One line; never raises, so a full disk cannot take the TUI down."""
    logger = _get()
    if logger is None:
        return
    try:
        logger.info(" ".join(str(message).split()), extra={"source": source})
    except Exception:
        pass
