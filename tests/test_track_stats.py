"""Tests for track_stats without a GPU, lab images or network."""

from track_stats import format_stats_table, summarize_tracks

# Track 1: frames 1-3. Track 2: frames 1, 3 (gap). Track 3: frame 3 only.
TEXT = (
    "1,1,0,0,10,10,0.9,-1,-1,-1\n"
    "1,2,0,0,10,10,0.9,-1,-1,-1\n"
    "2,1,0,0,10,10,0.9,-1,-1,-1\n"
    "3,1,0,0,10,10,0.9,-1,-1,-1\n"
    "3,2,0,0,10,10,0.9,-1,-1,-1\n"
    "3,3,0,0,10,10,0.9,-1,-1,-1\n"
)


def test_summarize_tracks_counts() -> None:
    stats = summarize_tracks(TEXT, short_len=2)
    assert stats["frames"] == 3
    assert stats["n_ids"] == 3
    assert stats["boxes_per_frame"] == 2.0
    assert stats["median_len"] == 2
    assert stats["short_frac"] == 1 / 3
    assert stats["gap_frac"] == 1 / 3


def test_summarize_tracks_empty() -> None:
    assert summarize_tracks("") == {}
    assert summarize_tracks("\n\n") == {}


def test_format_stats_table_one_line_per_run() -> None:
    stats = summarize_tracks(TEXT)
    table = format_stats_table([{"run": "video_2_botsort_c0.3_i0.5", **stats}])
    lines = table.splitlines()
    assert len(lines) == 2
    assert "video_2_botsort_c0.3_i0.5" in lines[1]
