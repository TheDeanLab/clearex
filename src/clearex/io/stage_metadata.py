#  Copyright (c) 2021-2026  The University of Texas Southwestern Medical Center.
#  All rights reserved.

"""Navigate multiposition stage metadata parsing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional
import json

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency guard
    yaml = None  # type: ignore[assignment]


def _looks_like_multiposition_header(row: Any) -> bool:
    """Return whether a row resembles a Navigate multiposition header."""
    if not isinstance(row, (list, tuple)) or not row:
        return False
    labels = {str(value).strip().upper() for value in row}
    return {"X", "Y", "Z"}.issubset(labels)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    """Return ``value`` as a finite float or ``default``."""
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if parsed != parsed:
        return float(default)
    return float(parsed)


def _payload_rows(payload: Any) -> Optional[list[Any]]:
    """Extract a row list from supported serialized multiposition payloads."""
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, Mapping):
        for key in ("positions", "MultiPositions", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return list(value)
    return None


def parse_multiposition_stage_rows(payload: Any) -> list[dict[str, float]]:
    """Parse Navigate multiposition rows into canonical stage-row dictionaries.

    Parameters
    ----------
    payload : Any
        Parsed ``multi_positions.yml`` or ``MultiPositions`` payload.

    Returns
    -------
    list of dict
        One dictionary per position with lowercase ``x``, ``y``, ``z``,
        ``theta``, and ``f`` float keys.
    """
    rows = _payload_rows(payload)
    if rows is None:
        return []

    header_index: dict[str, int] = {}
    if rows and _looks_like_multiposition_header(rows[0]):
        header = rows.pop(0)
        if isinstance(header, (list, tuple)):
            header_index = {
                str(value).strip().upper(): int(index)
                for index, value in enumerate(header)
            }

    parsed: list[dict[str, float]] = []
    for row in rows:
        if isinstance(row, Mapping):
            parsed.append(
                {
                    "x": _safe_float(row.get("x", row.get("X"))),
                    "y": _safe_float(row.get("y", row.get("Y"))),
                    "z": _safe_float(row.get("z", row.get("Z"))),
                    "theta": _safe_float(row.get("theta", row.get("THETA"))),
                    "f": _safe_float(row.get("f", row.get("F"))),
                }
            )
            continue

        if not isinstance(row, (list, tuple)):
            continue

        def _value(field: str, fallback_index: int) -> float:
            index = header_index.get(field, fallback_index)
            if index < 0 or index >= len(row):
                return 0.0
            return _safe_float(row[index])

        parsed.append(
            {
                "x": _value("X", 0),
                "y": _value("Y", 1),
                "z": _value("Z", 2),
                "theta": _value("THETA", 3),
                "f": _value("F", 4),
            }
        )
    return parsed


def _parse_serialized_text(text: str) -> Any:
    """Parse JSON or YAML text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise
        return yaml.safe_load(text)


def load_multiposition_stage_rows_from_directory(
    directory: Path,
) -> list[dict[str, float]]:
    """Load canonical stage rows from ``multi_positions.yml`` in ``directory``."""
    path = Path(directory).expanduser() / "multi_positions.yml"
    if not path.exists():
        return []
    payload = _parse_serialized_text(path.read_text(encoding="utf-8"))
    return parse_multiposition_stage_rows(payload)
