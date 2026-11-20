"""
UNI-T UTi Thermal Video Radiometric Extractor & Object Segmentation Utility.

Extracts calibrated per-pixel temperature arrays (°C) from screen-recorded
thermal camera videos by masking the UI overlay, reading the dynamic color-bar
bounds via Tesseract OCR, segmenting the foreground object, and resolving
target frames via audio-cue detection, peak temperature, or object placement.
"""

import os
import re
import shutil
import subprocess
import tempfile

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract as pyt
import scipy.io.wavfile as wavfile
from scipy.signal import spectrogram

# Auto-detect Tesseract executable across platforms, with Windows default fallback
_tesseract_path = shutil.which("tesseract") or os.environ.get("TESSERACT_CMD")
if _tesseract_path:
    pyt.pytesseract.tesseract_cmd = _tesseract_path
elif os.name == "nt":
    pyt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def coord_conv(a):
    """Convert 1-based (x, y) coordinates to 0-based (row, col) indices."""
    return (a[1] - 1, a[0] - 1)


def ttf(time_tuple, fps=30):
    """Convert a (minutes, seconds) tuple to a frame index."""
    return (time_tuple[0] * 60 + time_tuple[1]) * fps


def ftt(frame_no, fps=30):
    """Convert a frame index to a (minutes, seconds) tuple."""
    t = frame_no / fps
    return (int(t / 60), int(t % 60))


def get_frame(filepath, frame_no):
    """Read a single BGR frame at `frame_no` from a video file."""
    cap = cv.VideoCapture(filepath)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def showframe(fn_frame, title="frame", scale_factor=0.75):
    """Display an image frame in an OpenCV window until a key is pressed."""
    resized = cv.resize(fn_frame, None, fx=scale_factor, fy=scale_factor)
    cv.imshow(title, resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


def uimask(fn_frame):
    """Mask out the UNI-T UTi on-screen UI header and temperature scale bar."""
    masked_frame = fn_frame.copy()

    # Mask top status bar, upper/lower scale readouts, and vertical color bar
    cv.rectangle(masked_frame, (0, 0), (masked_frame.shape[1], 95), color=(0, 0, 0), thickness=-1)
    cv.rectangle(masked_frame, (538, 100), (668, 153), color=(0, 0, 0), thickness=-1)
    cv.rectangle(masked_frame, (538, 710), (668, 763), color=(0, 0, 0), thickness=-1)
    cv.rectangle(masked_frame, (625, 140), (668, 715), color=(0, 0, 0), thickness=-1)

    return masked_frame


def _parse_ocr_number(text):
    """Extract the first valid numeric temperature value from OCR text."""
    cleaned = text.strip()
    match_float = re.search(r"[-+]?\d+\.\d", cleaned)
    if match_float:
        return float(match_float.group(0))
    match = re.search(r"[-+]?\d+", cleaned)
    if match:
        val = float(match.group(0))
        if val > 350.0 and len(match.group(0)) >= 3:
            val = val / 10.0
        return val
    raise ValueError(f"Could not parse numeric temperature value from OCR text: '{text}'")


def _read_scale_val(thresh_img):
    """Run Tesseract OCR across multiple page-segmentation modes to read a scale bound."""
    for cfg in [
        "outputbase digits",
        "--psm 6 outputbase digits",
        "--psm 7 outputbase digits",
        "--psm 8 outputbase digits",
    ]:
        text = pyt.image_to_string(thresh_img, config=cfg)
        try:
            return _parse_ocr_number(text)
        except ValueError:
            continue
    raise ValueError("Could not parse scale number from OCR image")


def framescale(fn_frame):
    """
    Read the upper and lower temperature bounds from the UI scale bar via OCR.
    Returns (bar_up, bar_low, scale_per_intensity_unit).
    """
    frame_up = fn_frame[100:153, 538:668, :]
    _, thresh_up = cv.threshold(frame_up, 110, 255, cv.THRESH_BINARY)
    thresh_up[0:3, :] = 0  # Suppress top border noise from UI header

    frame_low = fn_frame[710:763, 538:668, :]
    _, thresh_low = cv.threshold(frame_low, 110, 255, cv.THRESH_BINARY)

    bar_up = _read_scale_val(thresh_up)
    bar_low = _read_scale_val(thresh_low)

    if bar_up <= bar_low or bar_up > 350.0 or bar_low < -20.0:
        raise ValueError(f"Physically implausible temperature scale: up={bar_up}, low={bar_low}")

    scale = (bar_up - bar_low) / 255.0
    return bar_up, bar_low, scale


def get_object_contour(cropped_frame, min_area=3000, max_area=250000):
    """
    Segment the primary foreground thermal object in the cropped viewport:
    - Excludes UI overlays and temperature scale bar.
    - Excludes objects touching frame borders (e.g., hands or tools entering frame).
    - Uses adaptive Otsu thresholding and morphological closing.
    - Filters candidates by area, solidity, and aspect ratio.
    """
    masked = uimask(cropped_frame)
    grey = cv.cvtColor(masked, cv.COLOR_RGB2GRAY)

    non_zero = grey[grey > 0]
    if len(non_zero) == 0:
        return None

    otsu_thresh, _ = cv.threshold(non_zero, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    blurred = cv.GaussianBlur(grey, (5, 5), 0)
    _, binary = cv.threshold(blurred, max(otsu_thresh, 35), 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid_candidates = []
    for c in contours:
        area = cv.contourArea(c)
        if area < min_area or area > max_area:
            continue

        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = float(area) / hull_area
        if solidity < 0.70:
            continue

        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.35 or aspect_ratio > 2.8:
            continue

        # Reject contours touching left/top/bottom frame boundaries
        if x <= 5 or y <= 95 or (y + h) >= 885:
            continue

        valid_candidates.append((area, c))

    if not valid_candidates:
        return None

    valid_candidates.sort(key=lambda x: x[0], reverse=True)
    return valid_candidates[0][1]


def detect_audio_beeps(video_path, target_freq_hz=2000, bandwidth_hz=100):
    """
    Detect a narrow-band audio cue (default: 2.0 kHz ± 100 Hz beep sequence)
    in the video's audio track using spectrogram band-energy ratio analysis.
    Returns a dict with 'detected', 'start_sec', 'end_sec', 'count', and 'beep_times'.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            wav_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.exists(wav_path):
            return {"detected": False, "start_sec": None, "end_sec": None, "count": 0, "beep_times": []}

        sr, data = wavfile.read(wav_path)
        if len(data) == 0:
            return {"detected": False, "start_sec": None, "end_sec": None, "count": 0, "beep_times": []}

        frequencies, times, Sxx = spectrogram(data, fs=sr, nperseg=512, noverlap=256)
        low_f = target_freq_hz - bandwidth_hz
        high_f = target_freq_hz + bandwidth_hz
        beep_band = (frequencies >= low_f) & (frequencies <= high_f)
        energy_band = np.sum(Sxx[beep_band, :], axis=0)

        broad_band = (frequencies >= 500) & (frequencies <= 6000)
        total_energy = np.mean(Sxx[broad_band, :], axis=0) + 1e-5
        ratio = energy_band / total_energy

        threshold = np.mean(ratio) + 2.5 * np.std(ratio)
        raw_hits = times[(ratio > threshold) & (energy_band > np.percentile(energy_band, 80))]

        if len(raw_hits) == 0:
            return {"detected": False, "start_sec": None, "end_sec": None, "count": 0, "beep_times": []}

        clustered = []
        current = [raw_hits[0]]
        for t in raw_hits[1:]:
            if t - current[-1] < 0.35:
                current.append(t)
            else:
                clustered.append((current[0], current[-1], np.mean(current)))
                current = [t]
        clustered.append((current[0], current[-1], np.mean(current)))

        valid_beeps = [c for c in clustered if (c[1] - c[0]) >= 0.03 or len(c) > 1]
        if not valid_beeps:
            valid_beeps = clustered

        sequences = []
        if valid_beeps:
            cur_seq = [valid_beeps[0]]
            for b in valid_beeps[1:]:
                if b[2] - cur_seq[-1][2] < 1.8:
                    cur_seq.append(b)
                else:
                    sequences.append(cur_seq)
                    cur_seq = [b]
            sequences.append(cur_seq)

        best_seq = max(sequences, key=len) if sequences else []
        if not best_seq:
            return {"detected": False, "start_sec": None, "end_sec": None, "count": 0, "beep_times": []}

        start_time = round(float(best_seq[0][0]), 2)
        end_time = round(float(best_seq[-1][1]), 2)
        beep_times = [round(float(b[2]), 2) for b in best_seq]

        return {
            "detected": True,
            "start_sec": start_time,
            "end_sec": end_time,
            "count": len(best_seq),
            "beep_times": beep_times,
        }
    except Exception:
        return {"detected": False, "start_sec": None, "end_sec": None, "count": 0, "beep_times": []}
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


def get_timed_temp_frame(
    video_path,
    delay_sec=5.0,
    beep_ref="end",
    search_duration_sec=60,
    return_contour=True,
    beep_info=None,
    target_freq_hz=2000,
):
    """
    Identify the audio-cue beep and resolve the measurement frame after `delay_sec`.
    Falls back gracefully to the maximum-temperature frame if no audio cue is found
    or if no stationary foreground object is detected post-cue.
    """
    crop_coords = (492, 69, 668, 892)
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()

    if beep_info is None:
        beep_info = detect_audio_beeps(video_path, target_freq_hz=target_freq_hz)

    frame_source = None
    target_frame = None
    resolved_frame = None
    object_contour = None

    if beep_info["detected"]:
        ref_time = beep_info["end_sec"] if beep_ref == "end" else beep_info["start_sec"]
        target_time = ref_time + delay_sec
        target_frame = int(round(target_time * fps))

        max_lookahead = min(total_frames - 1, target_frame + int(5.0 * fps))
        for f_idx in range(target_frame, max_lookahead + 1, 5):
            raw = get_frame(video_path, f_idx)
            if raw is not None:
                cf = raw[crop_coords[1]:crop_coords[1] + crop_coords[3], crop_coords[0]:crop_coords[0] + crop_coords[2], :]
                cnt = get_object_contour(cf)
                if cnt is not None:
                    resolved_frame = f_idx
                    object_contour = cnt
                    break

        if resolved_frame is not None:
            frame_source = "post beep frame"
        else:
            print(f"  Warning: Object was not stationary post-beep at frame {target_frame}. Falling back to max temp frame.")

    if resolved_frame is None:
        if not beep_info["detected"]:
            print("  Warning: No audio cue detected. Falling back to maximum temperature frame.")
        max_f, max_cnt = get_max_temp_frame(video_path, search_duration_sec=search_duration_sec, return_contour=True)
        resolved_frame = max_f
        object_contour = max_cnt
        frame_source = "max frame"

    rad_frame = extract_rad_frame(video_path, resolved_frame)
    mask = np.zeros(rad_frame.shape, dtype=np.uint8)
    cv.drawContours(mask, [object_contour], -1, 255, thickness=-1)
    object_pixels = rad_frame[mask == 255]

    mean_temp = float(np.mean(object_pixels)) if len(object_pixels) > 0 else 0.0
    std_temp = float(np.std(object_pixels)) if len(object_pixels) > 0 else 0.0

    result = {
        "frame_no": resolved_frame,
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "contour": object_contour,
        "frame_source": frame_source,
        "beep_info": beep_info,
        "target_frame": target_frame,
        "rad_frame": rad_frame,
    }

    if return_contour:
        return resolved_frame, object_contour, result
    return resolved_frame, result


def get_max_temp_frame(filepath, search_duration_sec=60, return_contour=False):
    """
    Scan the initial video window (e.g., first 60 seconds) at 1 Hz to locate
    the frame with the highest mean foreground object temperature.
    """
    crop_coords = (492, 69, 668, 892)
    cap = cv.VideoCapture(filepath)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    max_search_frame = min(total_frames, int(search_duration_sec * fps))

    grid_density = 10
    counter = 0
    records = []

    while counter < max_search_frame:
        ret, frame = cap.read()
        if not ret:
            break
        counter += 1

        if counter % 30 == 0:
            cropped = frame[crop_coords[1]:crop_coords[1] + crop_coords[3], crop_coords[0]:crop_coords[0] + crop_coords[2], :]

            object_contour = get_object_contour(cropped)
            if object_contour is None:
                continue

            try:
                b_up, b_low, scale = framescale(cropped)
            except Exception:
                continue

            x, y, w, h = cv.boundingRect(object_contour)
            gy = np.arange(y, y + h, grid_density)
            gx = np.arange(x, x + w, grid_density)
            gxv, gyv = np.meshgrid(gx, gy)
            grid_pts = np.column_stack((gxv.flat, gyv.flat)).astype(np.float64)

            inside = np.array([cv.pointPolygonTest(object_contour, tuple(pt), False) for pt in grid_pts]) > 0
            inside_pts = grid_pts[inside].astype(np.int32)

            if len(inside_pts) == 0:
                continue

            masked = uimask(cropped)
            grey = cv.cvtColor(masked, cv.COLOR_RGB2GRAY)

            y_coords = np.clip(inside_pts[:, 1] - 1, 0, grey.shape[0] - 1)
            x_coords = np.clip(inside_pts[:, 0] - 1, 0, grey.shape[1] - 1)
            pix_vals = grey[y_coords, x_coords]

            mean_val = float(np.mean(pix_vals))
            std_val = float(np.std(pix_vals))
            mean_temp = (scale * mean_val) + b_low
            std_temp = scale * std_val

            records.append({
                "frame_no": counter,
                "bar_up": b_up,
                "bar_low": b_low,
                "mean_temp": mean_temp,
                "std_temp": std_temp,
                "contour": object_contour,
            })

    cap.release()

    if not records:
        raise ValueError(f"Could not identify a stationary object or valid temperatures in {filepath}")

    df = pd.DataFrame(records)
    maxidx = df["mean_temp"].idxmax()
    best_row = df.loc[maxidx]
    maxframe = int(best_row["frame_no"])
    best_contour = best_row["contour"]

    if return_contour:
        return maxframe, best_contour
    return maxframe


def get_first_placed_frame(
    filepath,
    search_duration_sec=60,
    return_contour=True,
    beep_info=None,
    min_consecutive=3,
    step=5,
    target_freq_hz=2000,
):
    """
    Find the first frame where the target object is stationary in the viewport.
    If an audio cue is detected, searching begins after the cue ends.
    """
    crop_coords = (492, 69, 668, 892)
    cap = cv.VideoCapture(filepath)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()

    if beep_info is None:
        beep_info = detect_audio_beeps(filepath, target_freq_hz=target_freq_hz)

    start_sec = beep_info["end_sec"] if beep_info["detected"] else 0.0
    start_frame = int(start_sec * fps)
    max_search_frame = min(total_frames, start_frame + int(search_duration_sec * fps))

    consecutive = 0
    candidate_frame = None
    candidate_contour = None

    for f_idx in range(start_frame, max_search_frame, step):
        raw = get_frame(filepath, f_idx)
        if raw is None:
            break
        cropped = raw[crop_coords[1]:crop_coords[1] + crop_coords[3], crop_coords[0]:crop_coords[0] + crop_coords[2], :]
        cnt = get_object_contour(cropped)
        if cnt is not None:
            consecutive += 1
            if consecutive == 1:
                candidate_frame = f_idx
                candidate_contour = cnt
            if consecutive >= min_consecutive:
                break
        else:
            consecutive = 0
            candidate_frame = None
            candidate_contour = None

    if candidate_frame is None:
        print(f"  Warning: First placed frame could not be identified in {filepath}. Falling back to max temp frame.")
        max_f, max_cnt = get_max_temp_frame(filepath, search_duration_sec=search_duration_sec, return_contour=True)
        candidate_frame = max_f
        candidate_contour = max_cnt
        frame_source = "max frame (fallback)"
    else:
        frame_source = "first frame after placement"

    rad_frame = extract_rad_frame(filepath, candidate_frame)
    mask = np.zeros(rad_frame.shape, dtype=np.uint8)
    cv.drawContours(mask, [candidate_contour], -1, 255, thickness=-1)
    object_pixels = rad_frame[mask == 255]

    mean_temp = float(np.mean(object_pixels)) if len(object_pixels) > 0 else 0.0
    std_temp = float(np.std(object_pixels)) if len(object_pixels) > 0 else 0.0

    result = {
        "frame_no": candidate_frame,
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "contour": candidate_contour,
        "frame_source": frame_source,
        "rad_frame": rad_frame,
        "beep_info": beep_info,
    }

    if return_contour:
        return candidate_frame, candidate_contour, result
    return candidate_frame, result


def get_target_frame_by_mode(
    filepath,
    mode="post_beep",
    delay_sec=5.0,
    beep_ref="end",
    search_duration_sec=60,
    beep_info=None,
    target_freq_hz=2000,
):
    """
    Unified frame selector resolving the target measurement frame:
      - 'post_beep': fixed delay (e.g., 5.0 s) after detected audio cue, with max-temp fallback
      - 'max_temp': frame with highest mean object surface temperature
      - 'first_frame' (or 'first_placed'): first stationary object frame
    Returns (target_frame, contour, meta_dict).
    """
    if beep_info is None:
        beep_info = detect_audio_beeps(filepath, target_freq_hz=target_freq_hz)

    norm_mode = mode.lower().replace("-", "_")

    if norm_mode in ["post_beep", "timed", "post_beep_frame"]:
        return get_timed_temp_frame(
            filepath,
            delay_sec=delay_sec,
            beep_ref=beep_ref,
            search_duration_sec=search_duration_sec,
            return_contour=True,
            beep_info=beep_info,
            target_freq_hz=target_freq_hz,
        )
    elif norm_mode in ["first_frame", "first_placed", "first"]:
        return get_first_placed_frame(
            filepath,
            search_duration_sec=search_duration_sec,
            return_contour=True,
            beep_info=beep_info,
            target_freq_hz=target_freq_hz,
        )
    elif norm_mode in ["max_temp", "max", "max_frame"]:
        max_f, max_cnt = get_max_temp_frame(filepath, search_duration_sec=search_duration_sec, return_contour=True)
        rad_frame = extract_rad_frame(filepath, max_f)
        mask = np.zeros(rad_frame.shape, dtype=np.uint8)
        cv.drawContours(mask, [max_cnt], -1, 255, thickness=-1)
        object_pixels = rad_frame[mask == 255]
        mean_temp = float(np.mean(object_pixels)) if len(object_pixels) > 0 else 0.0
        std_temp = float(np.std(object_pixels)) if len(object_pixels) > 0 else 0.0
        meta = {
            "frame_no": max_f,
            "mean_temp": mean_temp,
            "std_temp": std_temp,
            "contour": max_cnt,
            "frame_source": "max frame",
            "beep_info": beep_info,
            "target_frame": max_f,
            "rad_frame": rad_frame,
        }
        return max_f, max_cnt, meta
    else:
        raise ValueError(f"Unknown frame mode: '{mode}'. Supported modes: 'post_beep', 'max_temp', 'first_frame'")


def get_frame_manual(filepath, frame_no):
    """
    Extract radiometric temperature data and object contour at an explicit frame index.
    Returns (frame_no, contour, meta_dict).
    """
    crop_coords = (492, 69, 668, 892)
    raw = get_frame(filepath, frame_no)
    if raw is None:
        raise ValueError(f"Could not read frame {frame_no} from {filepath}")

    cropped = raw[crop_coords[1]:crop_coords[1] + crop_coords[3], crop_coords[0]:crop_coords[0] + crop_coords[2], :]
    contour = get_object_contour(cropped)
    if contour is None:
        raise ValueError(f"Could not detect object contour in frame {frame_no} of {filepath}")

    rad_frame = extract_rad_frame(filepath, frame_no)
    mask = np.zeros(rad_frame.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 255, thickness=-1)
    object_pixels = rad_frame[mask == 255]

    mean_temp = float(np.mean(object_pixels)) if len(object_pixels) > 0 else 0.0
    std_temp = float(np.std(object_pixels)) if len(object_pixels) > 0 else 0.0

    meta = {
        "frame_no": frame_no,
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "contour": contour,
        "frame_source": "manual frame",
        "rad_frame": rad_frame,
    }

    return frame_no, contour, meta


def extract_rad_frame(filepath, frame_no):
    """
    Extract the calibrated radiometric temperature array (in °C) for `frame_no`.
    Uses adjacent-frame OCR fallback to handle single-frame video compression artifacts.
    """
    frame = get_frame(filepath, frame_no)
    if frame is None:
        raise ValueError(f"Could not read frame {frame_no} from {filepath}")

    crop_coords = (492, 69, 668, 892)
    cropped_frame = frame[crop_coords[1]:crop_coords[1] + crop_coords[3], crop_coords[0]:crop_coords[0] + crop_coords[2], :]

    try:
        _, bar_low, scale = framescale(cropped_frame)
    except Exception:
        scale = None
        for offset in [-1, 1, -2, 2, -3, 3, -5, 5, -10, 10, -30, 30]:
            try:
                adj = get_frame(filepath, frame_no + offset)
                if adj is not None:
                    adj_cropped = adj[crop_coords[1]:crop_coords[1] + crop_coords[3], crop_coords[0]:crop_coords[0] + crop_coords[2], :]
                    _, bar_low, scale = framescale(adj_cropped)
                    break
            except Exception:
                continue
        if scale is None:
            raise ValueError(f"Could not determine temperature scale for frame {frame_no} in {filepath}")

    masked_frame = uimask(cropped_frame)
    grey_frame = cv.cvtColor(masked_frame, cv.COLOR_RGB2GRAY)

    rad_frame = grey_frame * scale + bar_low
    return rad_frame


def isolate_object_frame(rad_frame, contour, bg_val=np.nan):
    """Mask out everything outside `contour`, replacing background with `bg_val`."""
    mask = np.zeros(rad_frame.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 255, thickness=-1)
    return np.where(mask == 255, rad_frame, bg_val)


def export_isolated_object_png(rad_frame, contour, out_filename, min_temp=28.0, max_temp=100.0):
    """Export an isolated foreground object thermal image on a black background."""
    mask = np.zeros(rad_frame.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 255, thickness=-1)

    fig, ax = plt.subplots(figsize=(6, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    masked_rad = np.ma.masked_where(mask != 255, rad_frame)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="black")

    ax.imshow(masked_rad, cmap=cmap, interpolation="nearest", vmin=min_temp, vmax=max_temp)
    ax.axis("off")

    plt.savefig(out_filename, bbox_inches="tight", pad_inches=0, dpi=500, facecolor="black")
    plt.close(fig)


def export_rad_frame_png(frame, out_filename, min_temp=28.0, max_temp=100.0, contour=None):
    """Export a full radiometric frame as a false-color inferno PNG."""
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(frame, cmap="inferno", interpolation=None, vmin=min_temp, vmax=max_temp)
    if contour is not None:
        c_poly = contour.reshape(-1, 2)
        ax.plot(
            np.append(c_poly[:, 0], c_poly[0, 0]),
            np.append(c_poly[:, 1], c_poly[0, 1]),
            color="cyan",
            linewidth=1.5,
        )
    ax.axis("off")
    plt.savefig(out_filename, bbox_inches="tight", pad_inches=0, dpi=500)
    plt.close(fig)


def export_rad_frame_csv(frame, filename):
    """Export a 2D radiometric temperature array (°C) to a headerless CSV file."""
    csv_name = filename + "_rad_data.csv" if not filename.endswith(".csv") else filename
    pd.DataFrame(frame).to_csv(csv_name, index=False, header=False)


def show_contours(fn_frame, contours):
    """Draw contours onto a copy of `fn_frame` for visual debugging."""
    contour_frame = fn_frame.copy()
    cv.drawContours(contour_frame, contours, -1, 255, 2)
    return contour_frame
