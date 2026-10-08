"""Tests for the pure helpers in sweep_tracking without a GPU, lab images or network."""

from pathlib import Path

import pytest

from sweep_tracking import (
    build_tracking_command,
    find_summary,
    format_table,
    one_at_a_time,
    parse_summary,
    run_name,
)

SUMMARY = "HOTA DetA AssA MOTA IDSW IDF1\n55.5 60.0 51.2 62.25 14 70.1\n"


def test_one_at_a_time_changes_one_value_and_dedups() -> None:
    configs = one_at_a_time(0.3, 0.5)
    assert configs[0] == (0.3, 0.5)
    assert len(configs) == len(set(configs)) == 5
    assert all((c != 0.3) + (i != 0.5) == 1 for c, i in configs[1:])


def test_one_at_a_time_base_outside_grid() -> None:
    assert one_at_a_time(0.25, 0.5) == [
        (0.25, 0.5), (0.15, 0.5), (0.3, 0.5), (0.5, 0.5), (0.25, 0.4), (0.25, 0.7),
    ]


def test_run_name() -> None:
    assert run_name("video_1", "bytetrack", 0.3, 0.5) == "video_1_bytetrack_c0.3_i0.5"


def test_parse_summary() -> None:
    assert parse_summary(SUMMARY) == {"HOTA": 55.5, "MOTA": 62.25, "IDF1": 70.1}


@pytest.mark.parametrize("text", ["", "HOTA MOTA\n1.0\n", "HOTA MOTA\n1.0 2.0\n"])
def test_parse_summary_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_summary(text)


def test_find_summary(tmp_path: Path) -> None:
    assert find_summary(tmp_path, "run1") is None
    target = tmp_path / "data/trackers/mot_challenge/X-train/run1/pedestrian_summary.txt"
    target.parent.mkdir(parents=True)
    target.write_text(SUMMARY)
    assert find_summary(tmp_path, "run1") == target


def test_build_tracking_command_flags() -> None:
    cmd = build_tracking_command(
        Path("img1"), "video_2", "botsort", 0.25, 0.5, Path("out"), "cuda:0", True, 150,
    )
    assert cmd[cmd.index("--tracker") + 1] == "botsort"
    assert "--save-video" in cmd
    assert cmd[cmd.index("--max-frames") + 1] == "150"
    full = build_tracking_command(
        Path("img1"), "video_2", "botsort", 0.25, 0.5, Path("out"), "cpu", False, 0,
    )
    assert "--save-video" not in full and "--max-frames" not in full


def test_format_table_marks_missing_scores() -> None:
    rows = [
        {"tracker": "bytetrack", "conf": 0.3, "iou": 0.5, "HOTA": 50.0, "MOTA": 60.0, "IDF1": 70.0},
        {"tracker": "ocsort", "conf": 0.3, "iou": 0.5},
    ]
    lines = format_table(rows).splitlines()
    assert len(lines) == 3
    assert "50.00" in lines[1]
    assert lines[2].split()[3:] == ["-", "-", "-"]
