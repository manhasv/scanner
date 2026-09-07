from __future__ import annotations

import curses
from pathlib import Path

"""Terminal-based file browser using curses."""

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic'}


def _list_entries(directory: Path) -> tuple[list[Path], list[Path]]:
    try:
        entries = list(directory.iterdir())
    except (PermissionError, FileNotFoundError):
        entries = []

    dirs = sorted(
        (e for e in entries if e.is_dir() and not e.name.startswith('.')),
        key=lambda e: e.name.lower(),
    )
    files = sorted(
        (e for e in entries if e.is_file() and e.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda e: e.name.lower(),
    )
    return dirs, files


def _browse(stdscr, start_dir: Path) -> str | None:
    curses.curs_set(0)
    stdscr.keypad(True)

    current_dir = start_dir
    selected = 0

    while True:
        dirs, files = _list_entries(current_dir)
        rows: list[tuple[str, str, Path | None]] = [("..", "up", current_dir.parent)]
        rows += [(f"{d.name}/", "dir", d) for d in dirs]
        rows += [(f.name, "file", f) for f in files]

        selected = max(0, min(selected, len(rows) - 1))

        stdscr.erase()
        height, width = stdscr.getmaxyx()
        stdscr.addnstr(0, 0, f" Select an image  -  {current_dir} ", max(width - 1, 0), curses.A_BOLD)

        visible_rows = max(height - 3, 1)
        top = max(0, selected - visible_rows + 1)
        for i, (label, _kind, _path) in enumerate(rows[top:top + visible_rows]):
            row_index = top + i
            attr = curses.A_REVERSE if row_index == selected else curses.A_NORMAL
            prefix = "> " if row_index == selected else "  "
            stdscr.addnstr(1 + i, 0, prefix + label, max(width - 1, 0), attr)

        footer = " up/down or j/k move | enter open/select | backspace/h up-dir | q cancel "
        stdscr.addnstr(height - 1, 0, footer, max(width - 1, 0), curses.A_DIM)
        stdscr.refresh()

        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = min(len(rows) - 1, selected + 1)
        elif key in (curses.KEY_BACKSPACE, 127, 8, ord('h'), curses.KEY_LEFT):
            current_dir = current_dir.parent
            selected = 0
        elif key in (curses.KEY_ENTER, 10, 13, curses.KEY_RIGHT, ord('l')):
            _label, kind, path = rows[selected]
            if kind in ("up", "dir"):
                current_dir = path
                selected = 0
            elif kind == "file":
                return str(path)
        elif key in (27, ord('q')):  # Esc or q
            return None


def select_image_file_tui(start_dir: Path | str | None = None) -> str | None:
    """Curses-based directory browser for picking an image file.

    Returns the selected file's path, or None if the user cancels (q/Esc).
    """
    start = Path(start_dir).resolve() if start_dir else Path.cwd()
    return curses.wrapper(_browse, start)
