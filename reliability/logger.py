"""
Structured JSON lines logger for MusicMind.
Each request produces one JSON object per log line in logs/musicmind.jsonl.
"""
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

# Resolve log directory relative to this file (musicmind/logs/) so that the
# logger works correctly regardless of which directory the process is launched from.
_LOG_DIR = Path(__file__).parent.parent / "logs"


class _JsonLineHandler(logging.Handler):
    """Writes log records as JSON objects, one per line."""

    def __init__(self) -> None:
        super().__init__()
        _LOG_DIR.mkdir(exist_ok=True)
        self._path = _LOG_DIR / "musicmind.jsonl"

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        standard_keys = set(logging.LogRecord.__dict__.keys()) | {
            "message", "asctime", "exc_info", "exc_text", "stack_info",
        }
        for k, v in record.__dict__.items():
            if k not in standard_keys and not k.startswith("_"):
                entry[k] = v

        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes JSON lines to logs/musicmind.jsonl."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(_JsonLineHandler())
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        logger.addHandler(console)
    return logger
