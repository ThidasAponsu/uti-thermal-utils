"""
Manual Frame / Timestamp Processor for UNI-T UTi Thermal Video Recordings.

Use this script when automatic audio-cue or contour tracking needs explicit
per-file timestamps (e.g., `(minutes, seconds)` converted via `ttf()`).
"""

import argparse
import os

import cv2 as cv
import pandas as pd

from uti_extractor import (
    export_isolated_object_png,
    export_rad_frame_csv,
    export_rad_frame_png,
    get_frame_manual,
    ttf,
)

# Map each video filename to an explicit frame number using `ttf((min, sec), fps=30)`.
# Example:
# FILE_FRAMES = {
#     "recording_01.mp4": ttf((0, 26)),
#     "recording_02.mp4": ttf((0, 36)),
# }
FILE_FRAMES = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manually extract radiometric frames at specified timestamps."
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing input .mp4 recordings (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs (default: same as --input-dir).",
    )
    parser.add_argument(
        "--summary-csv",
        default="summary_temperatures_manual.csv",
        help="Filename for the manual summary CSV (default: summary_temperatures_manual.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else video_dir
    os.makedirs(out_dir, exist_ok=True)

    if not FILE_FRAMES:
        print("FILE_FRAMES is empty. Populate FILE_FRAMES in manual_process.py before running.")
        return

    summary_records = []

    for filename, target_frame_no in FILE_FRAMES.items():
        file_stem = os.path.splitext(filename)[0]
        video_path = os.path.join(video_dir, filename)

        if not os.path.exists(video_path):
            print(f"Skipping {filename} (not found in {video_dir})")
            continue

        print("\n==========================================")
        print(f"Processing {filename} at frame {target_frame_no} ...")

        try:
            cap = cv.VideoCapture(video_path)
            fps = cap.get(cv.CAP_PROP_FPS) or 30.0
            cap.release()

            target_frame, contour, res_meta = get_frame_manual(video_path, target_frame_no)
            rad_frame = res_meta["rad_frame"]

            print(f"  Frame used: {res_meta['frame_source']} (frame {target_frame} at ~{target_frame / fps:.1f}s)")
            print(f"  Measured temperature: mean = {res_meta['mean_temp']:.2f}°C, std = {res_meta['std_temp']:.2f}°C")

            summary_records.append({
                "file_name": file_stem,
                "mean_temp": round(res_meta["mean_temp"], 2),
                "std_temp": round(res_meta["std_temp"], 2),
                "frame_mode": "manual",
                "frame_source": res_meta["frame_source"],
                "frame_no": target_frame,
                "time_sec": round(target_frame / fps, 2),
            })

            std_out = os.path.join(out_dir, f"{file_stem}_target_frame.png")
            export_rad_frame_png(rad_frame, std_out, contour=contour)
            print(f"  -> Saved full frame: {std_out}")

            isolated_out = os.path.join(out_dir, f"{file_stem}_isolated_object.png")
            export_isolated_object_png(rad_frame, contour, isolated_out)
            print(f"  -> Saved isolated object: {isolated_out}")

            csv_out = os.path.join(out_dir, f"{file_stem}_rad_data.csv")
            export_rad_frame_csv(rad_frame, csv_out)
            print(f"  -> Saved radiometric CSV: {csv_out}")

        except Exception as exc:
            print(f"  Error processing {filename}: {exc}")

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_csv_path = os.path.join(out_dir, args.summary_csv)
        summary_df.to_csv(summary_csv_path, index=False)
        print("\n==========================================")
        print("Batch Temperature Summary (manual):")
        print(summary_df.to_string(index=False))
        print(f"\n-> Saved manual summary CSV to: {summary_csv_path}")
        print("==========================================")

    print("\nManual batch processing complete.")


if __name__ == "__main__":
    main()
