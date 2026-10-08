#!/usr/bin/env python
"""Quét tracker và ngưỡng detector trên một video, mỗi lần chỉ đổi một số.

Script gọi ``run_tracking.py`` cho từng cấu hình. Với ``video_1`` chạy đủ frame và
có ``--trackeval-root``, script gọi thêm ``evaluate_practice.py`` rồi in bảng
HOTA / MOTA / IDF1. Các video khác chỉ sinh file kết quả và video xem thử để xem bằng mắt.

Ví dụ:
    python scripts/sweep_tracking.py \\
        --lab-data-root "$LAB_DATA" --video video_1 \\
        --trackers bytetrack botsort --device cuda:0 \\
        --trackeval-root ~/TrackEval --out runs/sweep
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
PRACTICE_VIDEO = "video_1"
DEFAULT_CONFS = (0.15, 0.3, 0.5)
DEFAULT_IOUS = (0.4, 0.5, 0.7)
SUMMARY_KEYS = ("HOTA", "MOTA", "IDF1")


def one_at_a_time(
    base_conf: float,
    base_iou: float,
    confs: Sequence[float] = DEFAULT_CONFS,
    ious: Sequence[float] = DEFAULT_IOUS,
) -> List[Tuple[float, float]]:
    """Liệt kê các cặp ``(conf, iou)`` mà mỗi cặp chỉ khác cấu hình gốc một số.

    Args:
        base_conf: ``conf`` gốc.
        base_iou: ``iou`` gốc.
        confs: Các giá trị ``conf`` cần thử, giữ ``iou`` gốc.
        ious: Các giá trị ``iou`` cần thử, giữ ``conf`` gốc.

    Returns:
        Danh sách không trùng lặp, cấu hình gốc đứng đầu.
    """
    configs = [(base_conf, base_iou)]
    configs += [(c, base_iou) for c in confs]
    configs += [(base_conf, i) for i in ious]
    seen = set()
    unique = []
    for cfg in configs:
        if cfg not in seen:
            seen.add(cfg)
            unique.append(cfg)
    return unique


def run_name(video: str, tracker: str, conf: float, iou: float) -> str:
    """Đặt tên thư mục cho một lần chạy.

    Args:
        video: Tên video, ví dụ ``video_1``.
        tracker: Tên tracker.
        conf: Ngưỡng confidence.
        iou: Ngưỡng IoU.

    Returns:
        Chuỗi dạng ``video_1_bytetrack_c0.3_i0.5``.
    """
    return f"{video}_{tracker}_c{conf:g}_i{iou:g}"


def parse_summary(text: str) -> Dict[str, float]:
    """Đọc file ``pedestrian_summary.txt`` của TrackEval.

    Args:
        text: Nội dung file: dòng đầu là tên cột, dòng hai là giá trị.

    Returns:
        Dict gồm ``HOTA``, ``MOTA``, ``IDF1`` (thang 0–100).

    Raises:
        ValueError: Khi file không đủ hai dòng, số cột không khớp, hoặc thiếu một trong ba cột.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("File tóm tắt TrackEval cần ít nhất hai dòng")
    names = lines[0].split()
    values = lines[1].split()
    if len(names) != len(values):
        raise ValueError(f"Số cột ({len(names)}) khác số giá trị ({len(values)})")
    table = dict(zip(names, values))
    missing = [key for key in SUMMARY_KEYS if key not in table]
    if missing:
        raise ValueError(f"Thiếu cột {missing} trong file tóm tắt")
    return {key: float(table[key]) for key in SUMMARY_KEYS}


def find_summary(trackeval_root: Path, name: str) -> Optional[Path]:
    """Tìm file tóm tắt TrackEval của một lần chấm.

    Args:
        trackeval_root: Thư mục gốc bản clone TrackEval.
        name: ``--run-name`` đã dùng khi chấm.

    Returns:
        Đường dẫn ``pedestrian_summary.txt``, hoặc None nếu chưa có.
    """
    pattern = f"data/trackers/mot_challenge/*/{name}/pedestrian_summary.txt"
    found = sorted(trackeval_root.glob(pattern))
    return found[0] if found else None


def build_tracking_command(
    source: Path,
    video: str,
    tracker: str,
    conf: float,
    iou: float,
    out_dir: Path,
    device: str,
    save_video: bool,
    max_frames: int,
) -> List[str]:
    """Dựng dòng lệnh gọi ``run_tracking.py``.

    Args:
        source: Thư mục ``img1`` của video.
        video: Tên video, dùng làm tên file ``.txt``.
        tracker: Tên tracker.
        conf: Ngưỡng confidence.
        iou: Ngưỡng IoU.
        out_dir: Thư mục xuất của lần chạy này.
        device: ``cpu`` hoặc ``cuda:0``.
        save_video: Có xuất video xem thử hay không.
        max_frames: Giới hạn frame; 0 là chạy toàn bộ.

    Returns:
        Danh sách đối số cho ``subprocess.run``.
    """
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_tracking.py"),
        "--source", str(source), "--seq-name", video,
        "--tracker", tracker, "--conf", str(conf), "--iou", str(iou),
        "--out", str(out_dir), "--device", device,
    ]
    if save_video:
        cmd.append("--save-video")
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    return cmd


def format_table(rows: Sequence[Dict[str, object]]) -> str:
    """Dựng bảng chữ căn cột cho kết quả quét.

    Args:
        rows: Mỗi phần tử có ``tracker``, ``conf``, ``iou`` và có thể có ``HOTA``, ``MOTA``, ``IDF1``.

    Returns:
        Chuỗi nhiều dòng. Ô thiếu số in ``-``.
    """
    lines = [f"{'tracker':11} {'conf':>5} {'iou':>5} {'HOTA':>6} {'MOTA':>6} {'IDF1':>6}"]
    for row in rows:
        cells = [f"{row[k]:6.2f}" if k in row else f"{'-':>6}" for k in SUMMARY_KEYS]
        lines.append(f"{row['tracker']:11} {row['conf']:5g} {row['iou']:5g} " + " ".join(cells))
    return "\n".join(lines)


def main() -> None:
    """Quét các cấu hình trên một video và in bảng tổng hợp."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab-data-root", required=True, type=Path)
    parser.add_argument("--video", required=True, help="video_1 … video_5")
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--base-conf", type=float, default=0.3)
    parser.add_argument("--base-iou", type=float, default=0.5)
    parser.add_argument(
        "--sweep-trackers", nargs="*", default=None,
        help="Tracker được quét conf/iou. Mặc định: chỉ chạy cấu hình gốc cho mọi tracker.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", help="Không xuất video xem thử")
    parser.add_argument("--trackeval-root", type=Path, default=None, help="Chỉ dùng cho video_1, chạy đủ frame")
    parser.add_argument("--out", type=Path, default=Path("runs/sweep"))
    args = parser.parse_args()

    source = args.lab_data_root / args.video / "img1"
    can_score = args.video == PRACTICE_VIDEO and args.trackeval_root and not args.max_frames
    if args.video == PRACTICE_VIDEO and args.trackeval_root and args.max_frames:
        print("Bỏ qua chấm số: --max-frames làm thiếu frame so với nhãn.")

    sweep_set = set(args.sweep_trackers or [])
    rows: List[Dict[str, object]] = []
    for tracker in args.trackers:
        if tracker in sweep_set:
            configs = one_at_a_time(args.base_conf, args.base_iou)
        else:
            configs = [(args.base_conf, args.base_iou)]
        for conf, iou in configs:
            name = run_name(args.video, tracker, conf, iou)
            out_dir = args.out / name
            cmd = build_tracking_command(
                source, args.video, tracker, conf, iou, out_dir,
                args.device, not args.no_video, args.max_frames,
            )
            print(f"\n=== {name} ===")
            subprocess.run(cmd, check=True)
            row: Dict[str, object] = {"tracker": tracker, "conf": conf, "iou": iou}
            if can_score:
                score_cmd = [
                    sys.executable, str(SCRIPTS_DIR / "evaluate_practice.py"),
                    "--trackeval-root", str(args.trackeval_root),
                    "--lab-data-root", str(args.lab_data_root),
                    "--submission", str(out_dir / f"{args.video}.txt"),
                    "--run-name", name,
                ]
                subprocess.run(score_cmd, check=True)
                summary = find_summary(args.trackeval_root, name)
                if summary is not None:
                    row.update(parse_summary(summary.read_text()))
            rows.append(row)

    print("\n" + format_table(rows))


if __name__ == "__main__":
    main()
