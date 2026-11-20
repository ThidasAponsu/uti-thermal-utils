"""
Automated Batch Processor for UNI-T UTi Thermal Video Recordings.

Discovers `.mp4` recordings in an input directory, resolves target frames via
audio-cue detection, peak temperature, or object placement, and exports:
  - Full radiometric false-color PNG (`<name>_target_frame.png`)
  - Isolated foreground object thermal PNG (`<name>_isolated_object.png`)
  - Calibrated per-pixel temperature matrix (`<name>_rad_data.csv`)
  - Batch summary table (`summary_temperatures.csv`)
"""

import argparse
import os

import cv2 as cv
import pandas as pd

from uti_extractor import (
    detect_audio_beeps,
    export_isolated_object_png,
    export_rad_frame_csv,
    export_rad_frame_png,
    get_max_temp_frame,
    get_target_frame_by_mode,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-process UNI-T UTi thermal camera MP4 recordings."
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
        "--mode",
        choices=["post_beep", "max_temp", "first_frame", "all"],
        default="post_beep",
        help="Frame selection mode (default: post_beep).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds after detected audio cue for 'post_beep' mode (default: 5.0).",
    )
    parser.add_argument(
        "--beep-ref",
        choices=["end", "start"],
        default="end",
        help="Reference edge of the audio-cue sequence (default: end).",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=2000,
        help="Target audio-cue center frequency in Hz (default: 2000).",
    )
    parser.add_argument(
        "--summary-csv",
        default="summary_temperatures.csv",
        help="Filename for the batch summary CSV (default: summary_temperatures.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else video_dir
    os.makedirs(out_dir, exist_ok=True)

    file_list = sorted(f for f in os.listdir(video_dir) if f.lower().endswith(".mp4"))
    if not file_list:
        print(f"No .mp4 files found in: {video_dir}")
        return

    print(f"Reading videos from: {video_dir}")
    print(f"Saving outputs to:   {out_dir}")
    print(f"Target frame mode:   {args.mode} (delay={args.delay}s, ref={args.beep_ref}, freq={args.freq}Hz)")

    summary_records = []

    for filename in file_list:
        file_stem = os.path.splitext(filename)[0]
        video_path = os.path.join(video_dir, filename)

        print("\n==========================================")
        print(f"Processing {filename} ...")

        try:
            cap = cv.VideoCapture(video_path)
            fps = cap.get(cv.CAP_PROP_FPS) or 30.0
            cap.release()

            # 1. Audio-cue detection
            beep_info = detect_audio_beeps(video_path, target_freq_hz=args.freq)
            if beep_info["detected"]:
                print(
                    f"  Audio cue detected: start = {beep_info['start_sec']}s, "
                    f"end = {beep_info['end_sec']}s ({beep_info['count']} beeps)"
                )
            else:
                print("  Audio cue detected: None")

            # 2. Maximum temperature frame (baseline reference)
            maxframe, _ = get_max_temp_frame(video_path, search_duration_sec=60, return_contour=True)
            print(f"  Maximum temperature frame: {maxframe} (~{maxframe / fps:.1f}s)")

            # 3. Frame resolution according to configured mode
            if args.mode == "all":
                _, _, meta_post = get_target_frame_by_mode(
                    video_path, mode="post_beep", delay_sec=args.delay,
                    beep_ref=args.beep_ref, beep_info=beep_info, target_freq_hz=args.freq,
                )
                _, _, meta_max = get_target_frame_by_mode(
                    video_path, mode="max_temp", beep_info=beep_info, target_freq_hz=args.freq,
                )
                _, _, meta_first = get_target_frame_by_mode(
                    video_path, mode="first_frame", beep_info=beep_info, target_freq_hz=args.freq,
                )

                target_frame = meta_post["frame_no"]
                contour = meta_post["contour"]
                rad_frame = meta_post["rad_frame"]
                frame_source = "post beep frame"

                print(f"  Frame used (default export): {frame_source} (frame {target_frame} at ~{target_frame / fps:.1f}s)")
                print(f"  Measured temperature: mean = {meta_post['mean_temp']:.2f}°C, std = {meta_post['std_temp']:.2f}°C")

                summary_records.append({
                    "file_name": file_stem,
                    "post_beep_mean": round(meta_post["mean_temp"], 2),
                    "post_beep_std": round(meta_post["std_temp"], 2),
                    "post_beep_frame": meta_post["frame_no"],
                    "max_temp_mean": round(meta_max["mean_temp"], 2),
                    "max_temp_std": round(meta_max["std_temp"], 2),
                    "max_temp_frame": meta_max["frame_no"],
                    "first_placed_mean": round(meta_first["mean_temp"], 2),
                    "first_placed_std": round(meta_first["std_temp"], 2),
                    "first_placed_frame": meta_first["frame_no"],
                })
            else:
                target_frame, contour, res_meta = get_target_frame_by_mode(
                    video_path,
                    mode=args.mode,
                    delay_sec=args.delay,
                    beep_ref=args.beep_ref,
                    beep_info=beep_info,
                    target_freq_hz=args.freq,
                )
                frame_source = res_meta["frame_source"]
                rad_frame = res_meta["rad_frame"]

                print(f"  Frame used: {frame_source} (frame {target_frame} at ~{target_frame / fps:.1f}s)")
                print(f"  Measured temperature: mean = {res_meta['mean_temp']:.2f}°C, std = {res_meta['std_temp']:.2f}°C")

                summary_records.append({
                    "file_name": file_stem,
                    "mean_temp": round(res_meta["mean_temp"], 2),
                    "std_temp": round(res_meta["std_temp"], 2),
                    "frame_mode": args.mode,
                    "frame_source": frame_source,
                    "frame_no": target_frame,
                    "time_sec": round(target_frame / fps, 2),
                    "beep_detected": beep_info["detected"],
                    "beep_start_sec": beep_info["start_sec"],
                    "beep_end_sec": beep_info["end_sec"],
                    "max_temp_frame": maxframe,
                })

            # 4. Export full frame PNG
            std_out = os.path.join(out_dir, f"{file_stem}_target_frame.png")
            export_rad_frame_png(rad_frame, std_out, contour=contour)
            print(f"  -> Saved full frame: {std_out}")

            # 5. Export isolated foreground object PNG
            isolated_out = os.path.join(out_dir, f"{file_stem}_isolated_object.png")
            export_isolated_object_png(rad_frame, contour, isolated_out)
            print(f"  -> Saved isolated object: {isolated_out}")

            # 6. Export radiometric CSV
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
        print(f"Batch Temperature Summary ({args.mode}):")
        print(summary_df.to_string(index=False))
        print(f"\n-> Saved batch summary CSV to: {summary_csv_path}")
        print("==========================================")

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
