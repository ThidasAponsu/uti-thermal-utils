"""
Interactive Radiometric CSV Viewer.

Loads a headerless 2D temperature CSV matrix (`*_rad_data.csv`) via CLI path
or a graphical file picker and renders it with interactive Min/Max color-scale
sliders.
"""

import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider


def _pick_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select Radiometric CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    root.destroy()
    return file_path


def plot_adjustable_radiometric_image(file_path=None):
    """Display a 2D radiometric CSV with interactive Min/Max temperature sliders."""
    if not file_path:
        file_path = _pick_file()

    if not file_path:
        print("No file selected. Exiting.")
        return

    try:
        df = pd.read_csv(file_path, header=None)
        image_data = df.to_numpy()

        data_min = float(np.nanmin(image_data))
        data_max = float(np.nanmax(image_data))

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2)

        img = ax.imshow(
            image_data,
            cmap="inferno",
            interpolation="nearest",
            vmin=data_min,
            vmax=data_max,
        )
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Temperature (°C)")

        ax.set_title("Radiometric Image")
        ax.set_xlabel("Columns (Pixels)")
        ax.set_ylabel("Rows (Pixels)")

        ax_min = plt.axes([0.15, 0.1, 0.7, 0.03])
        ax_max = plt.axes([0.15, 0.05, 0.7, 0.03])

        slider_min = Slider(ax_min, "Min Scale", data_min, data_max, valinit=data_min)
        slider_max = Slider(ax_max, "Max Scale", data_min, data_max, valinit=data_max)

        def update(_val):
            v_min = min(slider_min.val, slider_max.val - 0.01)
            v_max = max(slider_max.val, slider_min.val + 0.01)
            img.set_clim(vmin=v_min, vmax=v_max)
            fig.canvas.draw_idle()

        slider_min.on_changed(update)
        slider_max.on_changed(update)

        plt.show()

    except Exception as exc:
        print(f"An error occurred: {exc}")


if __name__ == "__main__":
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    plot_adjustable_radiometric_image(cli_path)
