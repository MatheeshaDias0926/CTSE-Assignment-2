"""
Centralized Logging System for Smart Factory MAS
==================================================
Provides structured, color-coded logging that traces every agent input,
tool call, and output during execution.

Usage:
    from logger import get_logger, log_agent_action, log_tool_call
    logger = get_logger("AgentName")
"""

import logging
import json
import os
from datetime import datetime
from typing import Any, Optional

from config import LOG_DIR


# ---------------------------------------------------------------------------
# ANSI Color Codes for Console Output
# ---------------------------------------------------------------------------
class Colors:
    """ANSI escape codes for colored terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GREY = "\033[90m"


# ---------------------------------------------------------------------------
# Custom Formatter with Color Support
# ---------------------------------------------------------------------------
class ColoredFormatter(logging.Formatter):
    """Adds ANSI colors to log levels for console readability."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GREY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        record.levelname = f"{color}{record.levelname:<8}{Colors.RESET}"
        record.name = f"{Colors.CYAN}{record.name}{Colors.RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------
_LOG_FILE: Optional[str] = None


def _get_log_file() -> str:
    """Return the shared log file path for the current session."""
    global _LOG_FILE
    if _LOG_FILE is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_FILE = os.path.join(LOG_DIR, f"mas_execution_{timestamp}.log")
    return _LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Create or retrieve a named logger with both console and file handlers.

    Args:
        name: Logger name (typically the agent or module name).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # Console handler (colored)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = ColoredFormatter(
        "%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    # File handler (plain text, verbose)
    file_handler = logging.FileHandler(_get_log_file(), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Structured Logging Helpers
# ---------------------------------------------------------------------------
def log_agent_action(
    logger: logging.Logger,
    agent_name: str,
    action: str,
    details: Any = None,
) -> dict:
    """
    Log a structured agent action and return the trace entry.

    Args:
        logger:     The logger instance to use.
        agent_name: Name of the agent performing the action.
        action:     Description of the action (e.g., 'invoke', 'route').
        details:    Additional context (dict, str, or any serialisable object).

    Returns:
        A dict representing the trace entry.
    """
    trace_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "details": _safe_serialize(details),
    }
    logger.info(
        f"{Colors.BOLD}[{agent_name}]{Colors.RESET} {action}"
        + (f" → {_truncate(str(details), 200)}" if details else "")
    )
    logger.debug(f"TRACE: {json.dumps(trace_entry, default=str)}")
    return trace_entry


def log_tool_call(
    logger: logging.Logger,
    tool_name: str,
    inputs: dict,
    output: Any,
) -> dict:
    """
    Log a tool invocation with its inputs and outputs.

    Args:
        logger:    The logger instance to use.
        tool_name: Name of the tool being called.
        inputs:    Dictionary of input arguments.
        output:    The tool's return value.

    Returns:
        A dict representing the trace entry.
    """
    trace_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "inputs": _safe_serialize(inputs),
        "output": _safe_serialize(output),
    }
    logger.info(
        f"{Colors.MAGENTA}🔧 TOOL [{tool_name}]{Colors.RESET}"
        f" inputs={_truncate(str(inputs), 150)}"
    )
    logger.info(
        f"{Colors.MAGENTA}🔧 TOOL [{tool_name}]{Colors.RESET}"
        f" output={_truncate(str(output), 300)}"
    )
    logger.debug(f"TOOL_TRACE: {json.dumps(trace_entry, default=str)}")
    return trace_entry


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------
def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate a string for display, appending '…' if trimmed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _safe_serialize(obj: Any) -> Any:
    """Convert an object to a JSON-safe representation."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    return str(obj)
