"""
time_tools.py

Time-related tools for LangChain / LangGraph.
Designed for LLM tool-calling with clear, constraint-oriented schemas.
"""

from typing import Union, Dict
from datetime import datetime, timedelta, timezone
from langchain.tools import tool, BaseTool

Number = Union[int, float]


@tool
def now_utc() -> str:
    """Get the current time in UTC (ISO 8601 format).

    Returns:
        Current UTC time as an ISO 8601 string.
    """
    return datetime.now(timezone.utc).isoformat()


@tool
def now_local() -> str:
    """Get the current local time (ISO 8601 format).

    Returns:
        Current local time as an ISO 8601 string.
    """
    return datetime.now().isoformat()


@tool
def parse_datetime(value: str) -> str:
    """Parse a datetime string and normalize it to ISO 8601 format.

    Args:
        value: A datetime string in ISO 8601 or common formats
               (e.g. '2025-01-01 10:30:00').

    Returns:
        Normalized ISO 8601 datetime string.
    """
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError("Invalid datetime format. Use ISO 8601 or 'YYYY-MM-DD HH:MM:SS'.")


@tool
def add_seconds(time: str, seconds: Number) -> str:
    """Add seconds to a datetime.

    Args:
        time: A datetime string in ISO 8601 format.
        seconds: Number of seconds to add (can be negative).
    """
    dt = datetime.fromisoformat(time)
    return (dt + timedelta(seconds=seconds)).isoformat()


@tool
def add_minutes(time: str, minutes: Number) -> str:
    """Add minutes to a datetime.

    Args:
        time: A datetime string in ISO 8601 format.
        minutes: Number of minutes to add (can be negative).
    """
    dt = datetime.fromisoformat(time)
    return (dt + timedelta(minutes=minutes)).isoformat()


@tool
def add_hours(time: str, hours: Number) -> str:
    """Add hours to a datetime.

    Args:
        time: A datetime string in ISO 8601 format.
        hours: Number of hours to add (can be negative).
    """
    dt = datetime.fromisoformat(time)
    return (dt + timedelta(hours=hours)).isoformat()


@tool
def add_days(time: str, days: Number) -> str:
    """Add days to a datetime.

    Args:
        time: A datetime string in ISO 8601 format.
        days: Number of days to add (can be negative).
    """
    dt = datetime.fromisoformat(time)
    return (dt + timedelta(days=days)).isoformat()


@tool
def diff_seconds(start: str, end: str) -> float:
    """Calculate the difference between two datetimes in seconds.

    Args:
        start: Start datetime (ISO 8601 format).
        end: End datetime (ISO 8601 format).

    Returns:
        Difference in seconds (end - start).
    """
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    return (end_dt - start_dt).total_seconds()


@tool
def diff_minutes(start: str, end: str) -> float:
    """Calculate the difference between two datetimes in minutes.

    Args:
        start: Start datetime (ISO 8601 format).
        end: End datetime (ISO 8601 format).

    Returns:
        Difference in minutes (end - start).
    """
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    return (end_dt - start_dt).total_seconds() / 60.0


@tool
def diff_hours(start: str, end: str) -> float:
    """Calculate the difference between two datetimes in hours.

    Args:
        start: Start datetime (ISO 8601 format).
        end: End datetime (ISO 8601 format).

    Returns:
        Difference in hours (end - start).
    """
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    return (end_dt - start_dt).total_seconds() / 3600.0


# Tool registry
tool_map: Dict[str, BaseTool] = {
    "now_utc": now_utc,
    "now_local": now_local,
    "parse_datetime": parse_datetime,
    "add_seconds": add_seconds,
    "add_minutes": add_minutes,
    "add_hours": add_hours,
    "add_days": add_days,
    "diff_seconds": diff_seconds,
    "diff_minutes": diff_minutes,
    "diff_hours": diff_hours,
}
