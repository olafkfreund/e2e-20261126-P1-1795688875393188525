"""Command-line entry point and word frequency helpers for wordfreq."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def count_words(
    text: str,
    *,
    min_len: int = 1,
    ignore_case: bool = True,
    top: int | None = None,
) -> list[tuple[str, int]]:
    """Count word occurrences in *text* and return `(word, count)` pairs.

    Words are runs of ASCII letters and digits. They are lowercased unless
    *ignore_case* is false, filtered to at least *min_len* characters, and
    ranked by `(-count, word)` so the output is deterministic. When *top* is
    given, only the first *top* pairs are returned.
    """
    if min_len < 1:
        msg = "min_len must be >= 1"
        raise ValueError(msg)
    if top is not None and top < 0:
        msg = "top must be >= 0"
        raise ValueError(msg)

    words = _TOKEN_PATTERN.findall(text)
    if ignore_case:
        words = [word.lower() for word in words]
    counts = Counter(word for word in words if len(word) >= min_len)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if top is not None:
        return ranked[:top]
    return ranked


def format_table(rows: Sequence[tuple[str, int]]) -> str:
    """Format `(word, count)` pairs as tab-separated lines."""
    return "\n".join(f"{word}\t{count}" for word, count in rows)


def _int_at_least(minimum: int) -> Callable[[str], int]:
    """Build an argparse type that accepts integers greater than or equal to *minimum*."""

    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            msg = f"invalid integer: {value!r}"
            raise argparse.ArgumentTypeError(msg) from None
        if number < minimum:
            msg = f"must be >= {minimum}"
            raise argparse.ArgumentTypeError(msg)
        return number

    return parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wordfreq",
        description="Deterministically count and rank words in text.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="file to read; reads stdin when omitted or '-'",
    )
    parser.add_argument(
        "--top",
        type=_int_at_least(0),
        default=None,
        help="show only the N most frequent words",
    )
    parser.add_argument(
        "--min-len",
        type=_int_at_least(1),
        default=1,
        help="ignore words shorter than N characters (default: 1)",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="count 'Apple' and 'apple' as different words",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a JSON array of {word, count} objects",
    )
    return parser


def _read_text(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wordfreq CLI and return the process exit code."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse has already written usage or error text to stderr.
        return exc.code if isinstance(exc.code, int) else 2

    source = args.path or "<stdin>"
    try:
        text = _read_text(args.path)
    except (OSError, UnicodeDecodeError) as exc:
        sys.stderr.write(f"wordfreq: error: cannot read {source}: {exc}\n")
        return 2

    rows = count_words(
        text,
        min_len=args.min_len,
        ignore_case=not args.case_sensitive,
        top=args.top,
    )
    if args.json:
        payload = [{"word": word, "count": count} for word, count in rows]
        sys.stdout.write(json.dumps(payload) + "\n")
    elif rows:
        sys.stdout.write(format_table(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
