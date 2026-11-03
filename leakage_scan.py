"""Regex-based data-leakage heuristics for ``data_validator.DataValidator``.

``LEAKAGE_PATTERNS`` maps an issue type to the line patterns that suggest it; adding a
pattern or a type needs no change to the scanner.  ``scan_lines`` is pure (lines in,
``(issue_type, line_number)`` hits out) so it can be tested without touching disk.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

LEAKAGE_PATTERNS: Dict[str, List[str]] = {
    "future_data": [
        r"\.shift\(-\d+\)",  # negative shift = looking forward
        r"\.rolling\(.*\)\.mean\(\).*\[-1\]",  # future-looking rolling
    ],
    "test_in_train": [
        r"test.*=.*\[:len",  # overlapping splits
        r"train.*=.*all_data",  # training on all data
    ],
    "unfair_split": [
        r"split.*random_state.*=.*None",  # non-reproducible split
        r"sample\(\).*frac",  # random sampling without seed
    ],
}


def scan_lines(
    lines: Iterable[str], patterns: Dict[str, List[str]] = LEAKAGE_PATTERNS
) -> List[Tuple[str, int]]:
    """Return ``(issue_type, 1-based line number)`` hits, ordered type -> pattern -> line."""
    lines = list(lines)
    hits: List[Tuple[str, int]] = []
    for issue_type, regexes in patterns.items():
        for pattern in regexes:
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    hits.append((issue_type, i + 1))
    return hits
