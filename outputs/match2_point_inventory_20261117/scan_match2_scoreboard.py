"""Build a human-review inventory of scoreboard displays; no OCR or point labels.

Frame numbers are zero based. Both video passes use grab/retrieve, without seek.
Only a temporary, reduced grayscale cache is kept for all sampled frames.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROGRESS_INTERVAL = 10_000
SHEET_LIMIT_BYTES = 500_000  # Strictly less than 500 decimal KB.
CROP_TARGET_BYTES = 100_000  # Approximate target; preserve resolution/quality.


@dataclass(frozen=True)
class Sample:
    frame: int
    diff: float
    brightness: float


@dataclass
class Segment:
    # Inclusive sample-index spans that passed the stability duration test.
    spans: list[tuple[int, int]] = field(default_factory=list)
    anchor_index: int = 0

    @property
    def first(self) -> int:
        return self.spans[0][0]

    @property
    def last(self) -> int:
        return self.spans[-1][1]

    def representative(self, samples: list[Sample]) -> int:
        """Nearest stable sample to the time midpoint; ties use the earlier one.

        In a merged interval, the absolute midpoint may fall in an occlusion.
        Restricting the choice to retained stable spans avoids that gap.
        """
        midpoint = (samples[self.first].frame + samples[self.last].frame) / 2
        step = samples[1].frame - samples[0].frame if len(samples) > 1 else 1
        best = self.first
        distance = float("inf")
        for first, last in self.spans:
            # Sample frames are uniformly spaced, including across discarded runs.
            offset = (midpoint - samples[first].frame) / step
            for candidate in (math.floor(offset) + first, math.ceil(offset) + first):
                candidate = max(first, min(last, candidate))
                key = (abs(samples[candidate].frame - midpoint), samples[candidate].frame)
                if key < (distance, samples[best].frame):
                    best, distance = candidate, key[0]
        return best


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "study/data/raw/match2.mp4")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--step", type=positive_int, default=15)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="actual-frame pixels; right/bottom are exclusive")
    parser.add_argument("--analysis-width", type=positive_int, default=320)
    parser.add_argument("--change-threshold", type=positive_float, default=None,
                        help="mean absolute gray difference in 0..255 units; default: automatic")
    parser.add_argument("--merge-threshold", type=positive_float, default=None,
                        help="representative-image difference; default: automatic")
    parser.add_argument("--min-stable-sec", type=positive_float, default=1.5)
    return parser.parse_args()


def write_csv(path: Path, columns: list[str], rows) -> None:
    # UTF-8 BOM makes the blank transcription template convenient in Excel.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean_absolute_diff(left: np.ndarray, right: np.ndarray) -> float:
    # uint8 subtraction would wrap around; absdiff performs saturated absolute diff.
    return float(cv2.absdiff(left, right).mean())


def progress(label: str, completed: int, target: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    speed = completed / max(elapsed, 1e-9)
    eta = f", ETA {max(0, target - completed) / speed / 60:.1f} min" if target else ""
    print(f"[{label}] {completed:,}/{target or '?'} frames, "
          f"{elapsed / 60:.1f} min, {speed:.1f} frames/s{eta}", flush=True)


def choose_crop(width: int, height: int, requested) -> tuple[int, int, int, int]:
    crop = tuple(requested) if requested is not None else (
        0, 0, round(width * 640 / 1920), round(height * 150 / 1080)
    )
    x1, y1, x2, y2 = crop
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid crop {crop} for actual frame {width}x{height}")
    return crop


def crop_frame(frame: np.ndarray, crop, dimensions) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) != dimensions:
        raise RuntimeError("Decoded video dimensions changed during scanning")
    x1, y1, x2, y2 = crop
    return frame[y1:y2, x1:x2]


def scan_samples(args, cache_path: Path, output: Path):
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {args.video}")
    samples: list[Sample] = []
    started = time.perf_counter()
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared_value = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        declared_count = round(declared_value) if math.isfinite(declared_value) else 0
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Invalid video FPS: {fps}")
        backend = cap.getBackendName()
        dimensions = crop = gray_shape = None
        previous = None
        frame_index = 0
        with cache_path.open("wb") as cache, (output / "samples.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "time_sec", "diff", "mean_brightness"])
            while cap.grab():
                if frame_index % args.step == 0:
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        raise RuntimeError(f"retrieve failed at F{frame_index}")
                    if dimensions is None:
                        dimensions = (frame.shape[1], frame.shape[0])
                        crop = choose_crop(*dimensions, args.crop)
                        crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
                        analysis_height = max(1, round(crop_height * args.analysis_width / crop_width))
                        gray_shape = (analysis_height, args.analysis_width)
                        print(f"Video {dimensions[0]}x{dimensions[1]}, {fps:.9f} fps; "
                              f"crop={crop}; analysis={gray_shape[1]}x{gray_shape[0]}", flush=True)
                    gray = cv2.cvtColor(crop_frame(frame, crop, dimensions), cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (gray_shape[1], gray_shape[0]), interpolation=cv2.INTER_AREA)
                    difference = 0.0 if previous is None else mean_absolute_diff(gray, previous)
                    brightness = float(gray.mean())
                    samples.append(Sample(frame_index, difference, brightness))
                    writer.writerow([frame_index, f"{frame_index / fps:.6f}",
                                     f"{difference:.8f}", f"{brightness:.8f}"])
                    cache.write(gray.tobytes(order="C"))
                    previous = gray
                frame_index += 1
                if frame_index % PROGRESS_INTERVAL == 0:
                    progress("pass 1/2", frame_index, declared_count, started)
        if not samples:
            raise RuntimeError("No sampled frames were decoded")
        # A failed grab is indistinguishable from EOF; compare with the header.
        # Do not silently turn an incomplete scan into a complete-match inventory.
        if declared_count > 0 and frame_index != declared_count:
            raise RuntimeError(f"Observed {frame_index} frames, header declares {declared_count}; "
                               "check the source/decoder before accepting the inventory")
        progress("pass 1/2 complete", frame_index, frame_index, started)
        return samples, {
            "fps": fps,
            "frame_count_declared": declared_count,
            "frame_count_observed": frame_index,
            "width": dimensions[0], "height": dimensions[1],
            "duration_sec": frame_index / fps,
            "crop_xyxy": list(crop),
            "analysis_shape_hw": list(gray_shape),
            "opencv_backend": backend,
            "pass_1_seconds": time.perf_counter() - started,
        }
    finally:
        cap.release()


def select_thresholds(samples, args):
    # The artificial diff=0 at F0 is excluded from every distribution statistic.
    values = np.asarray([sample.diff for sample in samples[1:]], dtype=np.float64)
    values = values[np.isfinite(values)]
    probabilities = [0, .05, .10, .25, .50, .75, .90, .95, .99, 1]
    quantiles = np.quantile(values, probabilities) if len(values) else np.zeros(len(probabilities))
    distribution = {f"q{round(p * 100):02d}": float(q) for p, q in zip(probabilities, quantiles)}
    median, iqr = distribution["q50"], distribution["q75"] - distribution["q25"]
    auto_change = max(.25, min(12.0, median + 1.5 * iqr))
    change = auto_change if args.change_threshold is None else args.change_threshold
    auto_merge = max(.05, min(change * .5, median + .5 * iqr))
    merge = auto_merge if args.merge_threshold is None else args.merge_threshold
    return change, merge, {
        "units": "mean absolute difference of uint8 gray pixels, range 0..255",
        "distribution_count": len(values),
        "excluded_first_sample": True,
        "diff_quantiles": distribution,
        "diff_median": median, "diff_iqr": iqr,
        "automatic_change_formula": "max(0.25, min(12.0, q50 + 1.5 * (q75 - q25)))",
        "automatic_merge_formula": "max(0.05, min(effective_change * 0.5, q50 + 0.5 * (q75 - q25)))",
        "automatic_change_value": auto_change, "automatic_merge_value": auto_merge,
        "change_threshold": change, "merge_threshold": merge,
        "change_source": "automatic" if args.change_threshold is None else "argument",
        "merge_source": "automatic" if args.merge_threshold is None else "argument",
        "comparison": "strictly less than threshold means stable/similar",
    }


def find_stable_segments(samples, fps, change_threshold, min_seconds):
    retained = []
    discarded_runs = 0
    first = 0
    # An edge diff[i] joins samples i-1 and i. A high edge splits these vertices.
    # The sentinel finalizes the last run and never reads samples[len(samples)].
    for index in range(1, len(samples) + 1):
        if index == len(samples) or samples[index].diff >= change_threshold:
            last = index - 1
            if (samples[last].frame - samples[first].frame) / fps >= min_seconds:
                middle = (first + last) // 2
                retained.append(Segment([(first, last)], anchor_index=middle))
            else:
                discarded_runs += 1
            first = index
    return retained, discarded_runs


def merge_segments(segments, samples, cache_path, gray_shape, threshold):
    if not segments:
        return [], []
    pixel_count = math.prod(gray_shape)
    merged: list[Segment] = []
    log = []
    with cache_path.open("rb") as cache:
        def read_gray(index):
            cache.seek(index * pixel_count)
            data = cache.read(pixel_count)
            if len(data) != pixel_count:
                raise RuntimeError(f"Incomplete gray cache at sample {index}")
            return np.frombuffer(data, dtype=np.uint8).reshape(gray_shape)

        current = segments[0]
        for following in segments[1:]:
            left_index = current.representative(samples)
            right_index = following.representative(samples)
            right = read_gray(right_index)
            difference = mean_absolute_diff(read_gray(left_index), right)
            # Also compare to the first raw interval's representative. This limits
            # chains of individually small changes drifting into a different score.
            anchor_difference = mean_absolute_diff(read_gray(current.anchor_index), right)
            accepted = difference < threshold and anchor_difference < threshold
            log.append({
                "left_rep_frame": samples[left_index].frame,
                "right_rep_frame": samples[right_index].frame,
                "anchor_frame": samples[current.anchor_index].frame,
                "gap_start_frame": samples[current.last].frame,
                "gap_end_frame": samples[following.first].frame,
                "representative_diff": difference, "anchor_diff": anchor_difference,
                "merged": accepted,
            })
            if accepted:
                current.spans.extend(following.spans)
            else:
                merged.append(current)
                current = following
        merged.append(current)
    return merged, log


def segment_rows(segments, samples, fps):
    rows = []
    for segment_id, segment in enumerate(segments, start=1):
        start, end = samples[segment.first].frame, samples[segment.last].frame
        representative = samples[segment.representative(samples)]
        rows.append({
            "segment_id": segment_id, "start_frame": start, "end_frame": end,
            "start_sec": round(start / fps, 6), "end_sec": round(end / fps, 6),
            "duration_sec": round((end - start) / fps, 6),
            "rep_frame": representative.frame,
            "mean_brightness": round(representative.brightness, 8),
        })
    return rows


def save_crop(image: Image.Image, stem: Path):
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=9, optimize=True)
    data = buffer.getvalue()
    suffix, encoding = ".png", "PNG compression=9, optimize=True"
    if len(data) > CROP_TARGET_BYTES:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90, subsampling=2, optimize=True)
        data = buffer.getvalue()
        suffix, encoding = ".jpg", "JPEG quality=90, subsampling=2, optimize=True"
    path = stem.with_suffix(suffix)
    path.write_bytes(data)
    return path, {"file": f"crops/{path.name}", "bytes": len(data),
                  "width": image.width, "height": image.height, "encoding": encoding,
                  "over_100kb_target": len(data) > CROP_TARGET_BYTES}


def extract_representatives(args, video_info, rows, output):
    if not rows:
        return {}, 0.0
    started = time.perf_counter()
    wanted = {row["rep_frame"]: row["segment_id"] for row in rows}
    target = max(wanted) + 1
    cap = cv2.VideoCapture(str(args.video))
    artifacts = {}
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot reopen video for representative extraction")
        for frame_index in range(target):
            if not cap.grab():
                raise RuntimeError(f"Second pass grab failed at F{frame_index}")
            if frame_index in wanted:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"Second pass retrieve failed at F{frame_index}")
                crop = crop_frame(frame, video_info["crop_xyxy"],
                                  (video_info["width"], video_info["height"]))
                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                segment_id = wanted[frame_index]
                _, info = save_crop(image, output / "crops" / f"segment_{segment_id:04d}")
                artifacts[segment_id] = info
            if (frame_index + 1) % PROGRESS_INTERVAL == 0:
                progress("pass 2/2", frame_index + 1, target, started)
        progress("pass 2/2 complete", target, target, started)
    finally:
        cap.release()
    if len(artifacts) != len(rows):
        raise RuntimeError("Representative count does not match segment count")
    return artifacts, time.perf_counter() - started


def timestamp(seconds):
    # Round once to tenths, so 59.96 becomes 01:00.0 rather than 00:60.0.
    tenths = round(seconds * 10)
    minutes, remainder = divmod(tenths, 600)
    return f"{minutes:02d}:{remainder // 10:02d}.{remainder % 10}"


def load_font(size):
    for name in ("C:/Windows/Fonts/consola.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_sheet_below_limit(image, path):
    # Retain the largest readable sheet that fits. Never cut off rows or crop digits.
    for scale in (1.0, .9, .8, .7, .6, .5):
        candidate = image if scale == 1 else image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
        for quality in (90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40):
            buffer = BytesIO()
            candidate.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=True)
            data = buffer.getvalue()
            if len(data) < SHEET_LIMIT_BYTES:
                path.write_bytes(data)
                return {"file": f"review_sheets/{path.name}", "bytes": len(data),
                        "width": candidate.width, "height": candidate.height,
                        "jpeg_quality": quality, "scale": scale}
    raise RuntimeError("Cannot fit a legible 24-row sheet below 500 KB; use a tighter --crop")


def make_review_sheets(rows, artifacts, fps, output):
    records = []
    font = load_font(23)
    small_font = load_font(19)
    label_width, image_width, margin = 225, 960, 12
    for page_start in range(0, len(rows), 24):
        group = rows[page_start:page_start + 24]
        first_info = artifacts[group[0]["segment_id"]]
        image_height = max(1, round(first_info["height"] * image_width / first_info["width"]))
        row_height = max(120, image_height) + margin
        sheet = Image.new("RGB", (label_width + image_width + margin, row_height * len(group)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(group):
            y = index * row_height
            draw.text((10, y + 8), f"segment_{row['segment_id']:04d}", fill="black", font=font)
            draw.text((10, y + 39), f"rep {timestamp(row['rep_frame'] / fps)}", fill="black", font=font)
            draw.text((10, y + 70), f"F{row['rep_frame']}", fill="black", font=small_font)
            draw.text((10, y + 97), f"from {timestamp(row['start_sec'])}", fill="black", font=small_font)
            info = artifacts[row["segment_id"]]
            with Image.open(output / info["file"]) as crop:
                tile = crop.convert("RGB").resize((image_width, image_height), Image.Resampling.LANCZOS)
            sheet.paste(tile, (label_width, y))
            draw.line((0, y + row_height - 1, sheet.width, y + row_height - 1), fill="#bbbbbb")
        page_number = page_start // 24 + 1
        record = save_sheet_below_limit(sheet, output / "review_sheets" / f"sheet_{page_number:02d}.jpg")
        record["segment_ids"] = [row["segment_id"] for row in group]
        records.append(record)
        print(f"Review sheet {page_number}: {record['bytes'] / 1000:.1f} KB", flush=True)
    return records


def main() -> int:
    args = parse_args()
    args.video = args.video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    # Protect a reviewer-filled template and avoid mixing runs/threshold settings.
    protected = ["samples.csv", "stable_segments.csv", "score_transcription_template.csv",
                 "metadata.json", "crops", "review_sheets"]
    collisions = [name for name in protected if (output / name).exists()]
    if collisions:
        raise FileExistsError(f"Existing run outputs: {', '.join(collisions)}. "
                              "Use --output-dir with a fresh folder; nothing was overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "crops").mkdir()
    (output / "review_sheets").mkdir()
    readme_source = HERE / "README.md"
    if output != HERE and readme_source.is_file() and not (output / "README.md").exists():
        (output / "README.md").write_bytes(readme_source.read_bytes())
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix=".scoreboard_cache_", dir=output) as temp:
        cache_path = Path(temp) / "samples_gray.bin"
        samples, video_info = scan_samples(args, cache_path, output)
        change, merge, threshold_info = select_thresholds(samples, args)
        print(f"Samples={len(samples):,}; change threshold={change:.6f}; "
              f"merge threshold={merge:.6f}", flush=True)
        raw_segments, discarded_runs = find_stable_segments(
            samples, video_info["fps"], change, args.min_stable_sec
        )
        # merge_segments extends spans in place; preserve the raw evidence first.
        raw_spans = [list(segment.spans[0]) for segment in raw_segments]
        segments, merge_log = merge_segments(
            raw_segments, samples, cache_path, tuple(video_info["analysis_shape_hw"]), merge
        )
        cache_bytes = cache_path.stat().st_size
    rows = segment_rows(segments, samples, video_info["fps"])
    write_csv(output / "stable_segments.csv",
              ["segment_id", "start_frame", "end_frame", "start_sec", "end_sec",
               "duration_sec", "rep_frame", "mean_brightness"], rows)
    template_columns = ["segment_id", "start_sec", "rep_frame", "game_score_near", "game_score_far",
                        "point_score_near", "point_score_far", "is_scoreboard_visible", "note"]
    write_csv(output / "score_transcription_template.csv", template_columns,
              ({key: row[key] if key in ("segment_id", "start_sec", "rep_frame") else ""
                for key in template_columns} for row in rows))
    artifacts, pass_2_seconds = extract_representatives(args, video_info, rows, output)
    sheets = make_review_sheets(rows, artifacts, video_info["fps"], output)
    stable_samples = sum(last - first + 1 for first, last in raw_spans)
    warnings = []
    if not rows:
        warnings.append("No stable segments. Inspect samples.csv and rerun with a new output folder.")
    over_target = sum(info["over_100kb_target"] for info in artifacts.values())
    if over_target:
        warnings.append(f"{over_target} original-resolution JPEG90 crops exceed the approximate 100 KB target.")
    if any(info["jpeg_quality"] < 60 or info["scale"] < .8 for info in sheets):
        warnings.append("Some sheets required strong compression; read original crops for ambiguous digits.")
    metadata = {
        "status": "complete", "started_utc": started_utc,
        "video_path": str(args.video), "video_size_bytes": args.video.stat().st_size,
        "output_dir": str(output), "opencv_version": cv2.__version__,
        **video_info,
        "step": args.step, "sample_interval_sec": args.step / video_info["fps"],
        "sample_count": len(samples), "min_stable_sec": args.min_stable_sec,
        "thresholds": threshold_info,
        "frame_convention": "zero-based; start/end are inclusive observed sample frames",
        "duration_convention": "(end_frame - start_frame) / fps; no extrapolation to transitions",
        "representative_convention": "stable sample nearest interval midpoint; earlier on ties",
        "mean_brightness_convention": "mean of reduced grayscale representative, range 0..255",
        "merge_convention": "adjacent retained intervals, representative AND fixed-anchor diff < merge_threshold; no gap-length limit",
        "merged_duration_includes_unstable_gaps": True,
        "raw_stable_segment_count": len(raw_spans), "discarded_short_run_count": discarded_runs,
        "stable_sample_count_before_merging": stable_samples,
        "stable_sample_fraction_before_merging": stable_samples / len(samples),
        "raw_stable_spans_frames": [[samples[first].frame, samples[last].frame] for first, last in raw_spans],
        "segment_count": len(rows), "sheet_count": len(sheets),
        "segment_stable_spans": [
            {"segment_id": row["segment_id"],
             "spans_frames": [[samples[first].frame, samples[last].frame] for first, last in segment.spans]}
            for row, segment in zip(rows, segments)
        ],
        "merge_log": merge_log,
        "crops": [artifacts[key] for key in sorted(artifacts)], "review_sheets": sheets,
        "crop_target_bytes_approximate": CROP_TARGET_BYTES,
        "sheet_limit_bytes_exclusive": SHEET_LIMIT_BYTES,
        "temporary_gray_cache_bytes_removed": cache_bytes,
        "pass_2_seconds": pass_2_seconds,
        "processing_seconds": time.perf_counter() - started,
        "warnings": warnings,
        "interpretation": "Display candidates only. Segment count is not point count; human video review is required.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
                                          + "\n", encoding="utf-8")
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)
    print(f"Done: {len(rows)} display segments, {len(sheets)} sheets; "
          f"{metadata['processing_seconds'] / 60:.1f} min. Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
