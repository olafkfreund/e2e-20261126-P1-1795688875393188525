"""Tests for the wordfreq package: counting helpers, stdin handling, and CLI behaviour."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wordfreq.cli import count_words, format_table, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _feed_stdin(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))


# --- count_words -----------------------------------------------------------


def test_count_words_basic_ranking() -> None:
    text = "Apple banana apple, cherry! Banana apple."
    assert count_words(text) == [("apple", 3), ("banana", 2), ("cherry", 1)]


def test_count_words_breaks_ties_alphabetically() -> None:
    assert count_words("pear apple mango") == [("apple", 1), ("mango", 1), ("pear", 1)]


def test_count_words_orders_by_count_before_word() -> None:
    assert count_words("apple zebra zebra") == [("zebra", 2), ("apple", 1)]


def test_count_words_result_does_not_depend_on_input_order() -> None:
    assert count_words("cherry apple banana") == count_words("banana cherry apple")


def test_count_words_lowercases_by_default() -> None:
    assert count_words("Apple APPLE apple") == [("apple", 3)]


def test_count_words_case_sensitive_keeps_case_distinct() -> None:
    assert count_words("Apple apple", ignore_case=False) == [("Apple", 1), ("apple", 1)]


def test_count_words_case_sensitive_ties_sort_by_code_point() -> None:
    # Uppercase letters have lower code points, so they sort before lowercase.
    assert count_words("apple Apple APPLE", ignore_case=False) == [
        ("APPLE", 1),
        ("Apple", 1),
        ("apple", 1),
    ]


def test_count_words_only_uses_ascii_letters_and_digits() -> None:
    # Underscores, apostrophes, and non-ASCII letters all split or drop tokens.
    assert count_words("foo_bar don't a1 b2 café") == [
        ("a1", 1),
        ("b2", 1),
        ("bar", 1),
        ("caf", 1),
        ("don", 1),
        ("foo", 1),
        ("t", 1),
    ]


def test_count_words_min_len_filters_short_words() -> None:
    assert count_words("a bb ccc bb", min_len=2) == [("bb", 2), ("ccc", 1)]


def test_count_words_top_limits_results() -> None:
    assert count_words("a a b b c", top=2) == [("a", 2), ("b", 2)]


def test_count_words_top_zero_returns_empty_list() -> None:
    assert count_words("a b", top=0) == []


def test_count_words_empty_text_returns_empty_list() -> None:
    assert count_words("") == []


@pytest.mark.parametrize("min_len", [0, -1])
def test_count_words_rejects_min_len_below_one(min_len: int) -> None:
    with pytest.raises(ValueError, match=r"^min_len must be >= 1$"):
        count_words("a", min_len=min_len)


def test_count_words_rejects_negative_top() -> None:
    with pytest.raises(ValueError, match=r"^top must be >= 0$"):
        count_words("a", top=-1)


# --- format_table ----------------------------------------------------------


def test_format_table_tsv() -> None:
    assert format_table([("apple", 3), ("banana", 2)]) == "apple\t3\nbanana\t2"


def test_format_table_empty_returns_empty_string() -> None:
    assert format_table([]) == ""


def test_format_table_has_no_trailing_newline() -> None:
    assert format_table([("a", 1)]) == "a\t1"


# --- CLI: input sources ----------------------------------------------------


def test_cli_reads_stdin_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "Apple apple pear\n")
    assert main([]) == 0
    assert capsys.readouterr().out == "apple\t2\npear\t1\n"


def test_cli_dash_path_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "x y x\n")
    assert main(["-"]) == 0
    assert capsys.readouterr().out == "x\t2\ny\t1\n"


def test_cli_empty_stdin_writes_nothing_for_tsv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "")
    assert main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_reads_file_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("to be or not to be that is the question", encoding="utf-8")
    rc = main([str(sample), "--min-len", "3", "--top", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines() == ["not\t1", "question\t1"]


# --- CLI: counting options -------------------------------------------------


def test_cli_counts_case_insensitively_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "Apple apple\n")
    assert main([]) == 0
    assert capsys.readouterr().out == "apple\t2\n"


def test_cli_case_sensitive_flag_keeps_case_distinct(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "Apple apple\n")
    assert main(["--case-sensitive"]) == 0
    assert capsys.readouterr().out == "Apple\t1\napple\t1\n"


def test_cli_tie_break_is_independent_of_input_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = []
    for text in ("cherry apple banana\n", "banana cherry apple\n"):
        _feed_stdin(monkeypatch, text)
        assert main([]) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] == "apple\t1\nbanana\t1\ncherry\t1\n"


def test_cli_top_and_min_len_from_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "a bb ccc ccc bb dddd\n")
    assert main(["--min-len", "2", "--top", "2"]) == 0
    assert capsys.readouterr().out == "bb\t2\nccc\t2\n"


# --- CLI: JSON output ------------------------------------------------------


def test_cli_json_output_is_array_with_trailing_newline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "apple apple pear\n")
    assert main(["--json"]) == 0
    assert capsys.readouterr().out == (
        '[{"word": "apple", "count": 2}, {"word": "pear", "count": 1}]\n'
    )


def test_cli_json_empty_input_prints_empty_array(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "")
    assert main(["--json"]) == 0
    assert capsys.readouterr().out == "[]\n"


def test_cli_json_honours_case_sensitive_and_top(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed_stdin(monkeypatch, "Apple apple apple\n")
    assert main(["--json", "--case-sensitive", "--top", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [{"word": "apple", "count": 2}]


# --- CLI: argument validation and errors -----------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["--top", "-1"],
        ["--top", "abc"],
        ["--top"],
        ["--min-len", "0"],
        ["--min-len", "x"],
        ["--bogus"],
    ],
)
def test_cli_rejects_invalid_arguments_with_exit_code_2(
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _feed_stdin(monkeypatch, "apple\n")
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""


@pytest.mark.parametrize("extra_args", [[], ["--json"]])
def test_cli_missing_file_returns_2_and_reports_on_stderr(
    tmp_path: Path,
    extra_args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.txt"
    assert main([str(missing), *extra_args]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing.txt" in captured.err


# --- python -m wordfreq ----------------------------------------------------


def _run_module(args: list[str], stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wordfreq", *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
        timeout=60,
    )


def test_python_m_wordfreq_reads_stdin_by_default() -> None:
    result = _run_module(["--top", "1"], stdin="the cat the dog\n")
    assert result.returncode == 0
    assert result.stdout == "the\t2\n"
    assert result.stderr == ""


def test_python_m_wordfreq_empty_stdin_writes_nothing() -> None:
    result = _run_module([], stdin="")
    assert result.returncode == 0
    assert result.stdout == ""


def test_python_m_wordfreq_json_ends_with_newline() -> None:
    result = _run_module(["--json"], stdin="b a\n")
    assert result.returncode == 0
    assert result.stdout == '[{"word": "a", "count": 1}, {"word": "b", "count": 1}]\n'


def test_python_m_wordfreq_missing_file_returns_2(tmp_path: Path) -> None:
    result = _run_module([str(tmp_path / "missing.txt")])
    assert result.returncode == 2
    assert result.stdout == ""
    assert "cannot read" in result.stderr


def test_python_m_wordfreq_invalid_argument_returns_2() -> None:
    result = _run_module(["--min-len", "0"], stdin="apple\n")
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr != ""
