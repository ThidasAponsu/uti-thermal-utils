# UTi Thermal Utils

A Python toolkit for extracting calibrated per-pixel radiometric temperature matrices (°C), segmenting foreground objects, and inspecting thermal frames from screen-recorded **UNI-T UTi** thermal camera videos.

## Features

- **OCR-Based Dynamic Scale Calibration:** Automatically reads upper and lower temperature bounds from the on-screen color bar using Tesseract OCR (with adjacent-frame fallback to handle compression artifacts).
- **UI Overlay Masking:** Strips on-screen HUD text, status headers, and scale bars before temperature calculation.
- **Foreground Object Segmentation:** Adaptive Otsu thresholding, morphological closing, and geometric filtering (area, solidity, aspect ratio, border exclusion) to isolate stationary objects in the viewport.
- **Audio-Cue Triggered Frame Selection:** Detects narrow-band audio cues (e.g., 2.0 kHz timer/equipment beeps) via spectrogram band-energy analysis to automatically resolve frames at a fixed post-cue delay, with graceful fallback to peak temperature.
- **Multiple Frame Resolution Modes:** Supports `post_beep` (audio-cue triggered), `max_temp` (peak mean object temperature), `first_frame` (first stationary placement), `all`, and explicit manual frame numbers.
- **Interactive Radiometric Viewer:** Tkinter + Matplotlib viewer for exploring exported `*_rad_data.csv` arrays with real-time `Min Scale` and `Max Scale` sliders.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `uti_extractor.py` | Core library for UI masking, OCR scale calibration, object segmentation, audio-cue detection, radiometric extraction, and PNG/CSV export. |
| `batch_process.py` | CLI batch processor that scans a folder of `.mp4` recordings, exports images/CSVs, and builds `summary_temperatures.csv`. |
| `manual_process.py` | Batch processor for specifying exact per-file timestamps/frame indices. |
| `radiometric_viewer.py` | Interactive GUI viewer with adjustable temperature scale sliders for exported `*_rad_data.csv` matrices. |

---

## Prerequisites

1. **Python 3.9+**
2. **Tesseract OCR**
   - **Windows:** Install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). If not on `PATH`, set the `TESSERACT_CMD` environment variable or install to the default `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   - **macOS:** `brew install tesseract`
   - **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
3. **FFmpeg** (required only for audio-cue detection in `post_beep` / `first_frame` modes):
   - Ensure `ffmpeg` is installed and available on your system `PATH`.

---

## Installation

```bash
git clone https://github.com/<your-username>/uti-thermal-utils.git
cd uti-thermal-utils
pip install -r requirements.txt
```

---

## Usage

### 1. Automated Batch Processing (`batch_process.py`)

Process all `.mp4` files in a directory using audio-cue triggered frame selection (default: 5.0 s after a 2.0 kHz audio cue ends):

```bash
python batch_process.py --input-dir ./recordings --mode post_beep --delay 5.0
```

Other modes:
```bash
# Select the frame with the highest mean foreground object temperature
python batch_process.py --input-dir ./recordings --mode max_temp

# Select the first frame where the object becomes stationary
python batch_process.py --input-dir ./recordings --mode first_frame

# Evaluate all three modes and record them side-by-side in summary_temperatures.csv
python batch_process.py --input-dir ./recordings --mode all

# Custom audio-cue center frequency (e.g., 1500 Hz) and separate output folder
python batch_process.py --input-dir ./recordings --output-dir ./outputs --freq 1500
```

#### Generated Outputs
For each `<name>.mp4`, the processor writes:
- `<name>_target_frame.png` — Full false-color (`inferno`) thermal frame with detected object contour overlay.
- `<name>_isolated_object.png` — Isolated foreground object on a black background.
- `<name>_rad_data.csv` — Headerless 2D matrix of calibrated temperatures in °C.
- `summary_temperatures.csv` — Aggregated mean/std temperatures, frame indices, timestamps, and audio-cue metadata across the batch.

---

### 2. Manual Frame Processing (`manual_process.py`)

For recordings that require exact timestamps, populate `FILE_FRAMES` in `manual_process.py` using `ttf((minutes, seconds))`:

```python
FILE_FRAMES = {
    "recording_01.mp4": ttf((0, 26)),
    "recording_02.mp4": ttf((0, 36)),
}
```

Then run:
```bash
python manual_process.py --input-dir ./recordings
```

---

### 3. Interactive Radiometric CSV Viewer (`radiometric_viewer.py`)

Open a file dialog to select any `*_rad_data.csv` file, or pass the path directly:

```bash
python radiometric_viewer.py
# or
python radiometric_viewer.py ./outputs/recording_01_rad_data.csv
```

---

## Viewport Configuration Note

`uti_extractor.py` assumes a screen-recorded UNI-T UTi viewport cropped at `crop_coords = (492, 69, 668, 892)` with scale-bar OCR regions at `[100:153, 538:668]` and `[710:763, 538:668]`. If your recording resolution or window layout differs, update `crop_coords`, `uimask()`, and `framescale()` in `uti_extractor.py`.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in academic or technical work, please cite it using the **"Cite this repository"** button on GitHub or via [`CITATION.cff`](CITATION.cff):

```bibtex
@software{aponsu_2026_uti_thermal_utils,
  author       = {Aponsu, G. M. T. P.},
  title        = {UTi Thermal Utils: Radiometric Video Extractor \& Object Segmentation Toolkit},
  year         = {2026},
  version      = {1.0.0},
  license      = {MIT}
}
```
