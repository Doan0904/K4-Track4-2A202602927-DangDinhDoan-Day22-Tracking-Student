#!/usr/bin/env python
"""Tóm tắt độ ổn định của file kết quả tracking, không cần nhãn.

Với video không có nhãn, các số này chỉ là chỉ báo phụ để chọn video nào cần xem kỹ.
Chúng không thay cho việc xem video: ít ID có thể do tracker tốt, cũng có thể do
detector bỏ sót nhiều người.

Ví dụ:
    python scripts/track_stats.py --runs-dir runs/eye runs/eye_sweep
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

SHORT_TRACK_FRAMES = 15


def summarize_tracks(text: str, short_len: int = SHORT_TRACK_FRAMES) -> Dict[str, float]:
    """Tóm tắt file kết quả định dạng MOT (``frame,id,x,y,w,h,conf,...``).

    Args:
        text: Nội dung file ``video_N.txt``.
        short_len: Track ngắn hơn số frame này bị coi là ngắn.

    Returns:
        Dict gồm ``frames`` (frame cuối có hộp), ``boxes_per_frame``, ``n_ids``,
        ``median_len`` (độ dài trung vị của track), ``short_frac`` (tỉ lệ track ngắn)
        và ``gap_frac`` (tỉ lệ track bị đứt quãng rồi nối lại). Dict rỗng khi không có hộp.
    """
    track_frames: Dict[int, List[int]] = defaultdict(list)
    n_boxes = 0
    last_frame = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        frame, track_id = int(parts[0]), int(parts[1])
        track_frames[track_id].append(frame)
        n_boxes += 1
        last_frame = max(last_frame, frame)
    if not track_frames:
        return {}
    lengths = [len(f) for f in track_frames.values()]
    gaps = sum(1 for f in track_frames.values() if len(f) < max(f) - min(f) + 1)
    n_ids = len(track_frames)
    return {
        "frames": last_frame,
        "boxes_per_frame": n_boxes / last_frame,
        "n_ids": n_ids,
        "median_len": statistics.median(lengths),
        "short_frac": sum(1 for n in lengths if n < short_len) / n_ids,
        "gap_frac": gaps / n_ids,
    }


def format_stats_table(rows: Sequence[Dict[str, object]]) -> str:
    """Dựng bảng chữ cho các lần chạy.

    Args:
        rows: Mỗi phần tử có ``run`` (tên lần chạy) và các khóa của ``summarize_tracks``.

    Returns:
        Chuỗi nhiều dòng, một dòng mỗi lần chạy.
    """
    header = f"{'lần chạy':40} {'hộp/fr':>7} {'ID':>5} {'trung vị':>9} {'ngắn%':>6} {'đứt%':>6}"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['run']:40} {row['boxes_per_frame']:7.1f} {row['n_ids']:5d} "
            f"{row['median_len']:9.1f} {100 * row['short_frac']:6.0f} {100 * row['gap_frac']:6.0f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Đọc mọi ``*/video_N.txt`` trong các thư mục chạy và in bảng tóm tắt."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", nargs="+", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for runs_dir in args.runs_dir:
        for txt in sorted(runs_dir.glob("*/video_*.txt")):
            stats = summarize_tracks(txt.read_text())
            if stats:
                rows.append({"run": txt.parent.name, **stats})
    print(format_stats_table(rows))


if __name__ == "__main__":
    main()
