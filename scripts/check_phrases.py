"""Fail when a tracked text file breaks the repo's writing rules.

The rules are in CLAUDE.md: plain prose, no em dashes, and no appeals to candour in place
of evidence. Files written before the rules are listed in LEGACY and left alone.

    python scripts/check_phrases.py            # every tracked file
    python scripts/check_phrases.py a.md b.py  # the files given (pre-commit)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Spelled indirectly so this file passes its own check.
RULES = {
    "em dash": re.compile(chr(0x2014)),
    "candour word": re.compile("hon" + "est", re.IGNORECASE),
}
LEGACY = ("notebooks/", "docs/da/", "src/heat/da/", "scripts/fetch_da_", "tests/test_da_", "app.py")
TEXT = {".py", ".md", ".yml", ".yaml", ".toml", ".json", ".ts", ".tsx", ".css", ".html", ".sh", ".txt"}


def violations(path: str, text: str) -> list[str]:
    """One line per rule broken, as `path:line: rule`."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for name, pattern in RULES.items():
            if pattern.search(line):
                out.append(f"{path}:{n}: {name}")
    return out


def checked(path: str) -> bool:
    return Path(path).suffix in TEXT and not path.startswith(LEGACY)


def main(argv: list[str]) -> int:
    paths = argv or subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split("\n")
    found: list[str] = []
    for p in filter(checked, paths):
        f = Path(p)
        if f.is_file():
            found += violations(p, f.read_text(errors="replace"))
    print("\n".join(found))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
