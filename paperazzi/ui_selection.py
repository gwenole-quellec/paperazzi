"""Resolve clicked Streamlit dataframe cells to persistent database IDs.

A native ``single-cell`` selection turns any cell into an editor shortcut and
removes Streamlit's checkbox column. Cell selections contain a positional row
index, not a database identifier, so the mapping must use the rendered rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def selected_row_identifier(
    event_state: dict[str, Any] | None,
    record_ids: Sequence[int],
) -> int | None:
    """Return the persistent record ID corresponding to the clicked cell.

    Both cell selections (the current UI) and older row selections are
    supported. Streamlit indices refer to the original dataframe positions,
    even when the user changes the visual sort order of its columns.

    Missing, malformed, or out-of-range events do not change the editor.
    """
    if not event_state:
        return None

    selection = event_state.get("selection") or {}
    cells = selection.get("cells") or []
    rows = selection.get("rows") or []

    if cells:
        cell = cells[0]
        if not isinstance(cell, (tuple, list)) or len(cell) != 2:
            return None
        position = cell[0]
    elif rows:
        position = rows[0]
    else:
        return None

    if type(position) is not int or not 0 <= position < len(record_ids):
        return None

    return record_ids[position]
