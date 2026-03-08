import json
from dataclasses import dataclass, asdict
from typing import List, Optional
import os
import numpy as np
import pandas as pd
import imageio.v2 as iio
import dask.array as da
from dask import delayed
import pims
import cv2

import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QFileDialog, QSpinBox
)
from napari_video.napari_video import VideoReaderNP


# ----------------------------
# helpers
# ----------------------------
def frame_to_timecode(frame_idx: int, fps: float) -> str:
    sec = frame_idx / fps
    mm = int(sec // 60)
    ss = sec % 60
    return f"{mm:02d}:{ss:06.3f}"

def load_video(viewer, path):
    vr = VideoReaderNP(path)
    viewer.add_image(vr, name="video")
    return vr

def load_individuals_from_h5(h5_path: str) -> List[str]:
    df = pd.read_hdf(h5_path)
    if not isinstance(df.columns, pd.MultiIndex):
        return []
    if "individuals" not in df.columns.names:
        return []
    inds = list(df.columns.get_level_values("individuals").unique())
    inds = [str(i) for i in inds if str(i) != "single"]
    return inds


@dataclass
class Edit:
    type: str                 # "swap" or "blank"
    t0_frame: int
    t1_frame: int
    t0: str
    t1: str
    a: Optional[str] = None
    b: Optional[str] = None
    ind: Optional[str] = None


# ----------------------------
# UI Widget
# ----------------------------
class AnnotatorWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self.video = None
        self.video_path = None
        self.dlc_df = None
        self.fps = 30.0
        self.individuals: List[str] = []
        self.edits: List[Edit] = []

        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None

        # ---- UI ----
        layout = QVBoxLayout()

        # Load buttons
        btn_row = QHBoxLayout()
        self.btn_load_video = QPushButton("Load video (.mp4)")
        self.btn_load_h5 = QPushButton("Load DLC .h5 (optional)")
        btn_row.addWidget(self.btn_load_video)
        btn_row.addWidget(self.btn_load_h5)
        layout.addLayout(btn_row)

        # FPS
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_box = QSpinBox()
        self.fps_box.setRange(1, 240)
        self.fps_box.setValue(30)
        fps_row.addWidget(self.fps_box)
        layout.addLayout(fps_row)

        # Type
        self.type_box = QComboBox()
        self.type_box.addItems(["swap", "blank"])
        layout.addWidget(QLabel("Edit type"))
        layout.addWidget(self.type_box)

        # Individuals selectors
        layout.addWidget(QLabel("Swap A / Swap B"))
        swap_row = QHBoxLayout()
        self.a_box = QComboBox()
        self.b_box = QComboBox()
        swap_row.addWidget(self.a_box)
        swap_row.addWidget(self.b_box)
        layout.addLayout(swap_row)

        layout.addWidget(QLabel("Blank individual"))
        self.ind_box = QComboBox()
        layout.addWidget(self.ind_box)

        # Start/end/commit
        mark_row = QHBoxLayout()
        self.btn_set_start = QPushButton("Set START")
        self.btn_set_end = QPushButton("Set END")
        mark_row.addWidget(self.btn_set_start)
        mark_row.addWidget(self.btn_set_end)
        layout.addLayout(mark_row)

        commit_row = QHBoxLayout()
        self.btn_commit = QPushButton("Commit edit")
        self.btn_undo = QPushButton("Undo")
        commit_row.addWidget(self.btn_commit)
        commit_row.addWidget(self.btn_undo)
        layout.addLayout(commit_row)

        save_row = QHBoxLayout()
        self.btn_save = QPushButton("Save JSON")
        save_row.addWidget(self.btn_save)
        layout.addLayout(save_row)

        # Status
        self.status = QLabel("No video loaded.")
        layout.addWidget(self.status)

        self.setLayout(layout)

        # ---- Connections ----
        self.btn_load_video.clicked.connect(self.load_video_clicked)
        self.btn_load_h5.clicked.connect(self.load_h5_clicked)
        self.fps_box.valueChanged.connect(self.on_fps_changed)
        self.type_box.currentTextChanged.connect(self.refresh_visibility)

        self.btn_set_start.clicked.connect(self.set_start)
        self.btn_set_end.clicked.connect(self.set_end)
        self.btn_commit.clicked.connect(self.commit)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_save.clicked.connect(self.save_json)

        self.refresh_visibility()

    # ----------------------------
    # callbacks
    # ----------------------------
    def on_fps_changed(self, v: int):
        self.fps = float(v)
        self.update_status()

    def load_video_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Video (*.mp4 *.avi *.mov)"
        )

        if not path:
            return

        self.video_path = path

        self.viewer.layers.clear()

        self.video = load_video(self.viewer, path)

        self.viewer.dims.set_point(0, 0)

        self.update_status()

    def load_h5_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DLC h5", "", "H5 (*.h5 *.hdf5)"
        )
        if not path:
            return

        # Load dataframe and keep it
        self.dlc_df = pd.read_hdf(path)

        # Get individuals
        inds = load_individuals_from_h5(path)
        self.individuals = inds if inds else []

        # Update dropdowns
        self.a_box.clear()
        self.b_box.clear()
        self.ind_box.clear()

        if self.individuals:
            self.a_box.addItems(self.individuals)
            self.b_box.addItems(self.individuals)
            self.ind_box.addItems(self.individuals)
            if len(self.individuals) > 1:
                self.b_box.setCurrentIndex(1)

        # Remove old point layers if they exist
        for layer_name in list(self.viewer.layers):
            if layer_name.name.startswith("pts_"):
                self.viewer.layers.remove(layer_name)

        # ---- Display one bodypart per individual ----
        scorer = self.dlc_df.columns.get_level_values("scorer")[0]

        # choose the bodypart you want to visualize
        bodypart_to_show = "middle"

        for ind in self.individuals:
            x_col = (scorer, ind, bodypart_to_show, "x")
            y_col = (scorer, ind, bodypart_to_show, "y")

            if x_col not in self.dlc_df.columns or y_col not in self.dlc_df.columns:
                print(f"Skipping {ind}: bodypart '{bodypart_to_show}' not found.")
                continue

            x = self.dlc_df[x_col].values
            y = self.dlc_df[y_col].values

            # Napari points for time series must be (frame, y, x)
            frames = np.arange(len(x))
            valid = ~(np.isnan(x) | np.isnan(y))

            points = np.column_stack([
                frames[valid],
                y[valid],
                x[valid],
            ])

            self.viewer.add_points(
                points,
                name=f"pts_{ind}",
                size=6,
            )

        self.update_status()

    def refresh_visibility(self):
        etype = self.type_box.currentText()
        is_swap = (etype == "swap")
        self.a_box.setEnabled(is_swap)
        self.b_box.setEnabled(is_swap)
        self.ind_box.setEnabled(not is_swap)

    def current_frame(self) -> int:
        # dim 0 is time axis
        return int(self.viewer.dims.point[0])

    def set_start(self):
        if self.video is None:
            return
        self.start_frame = self.current_frame()
        self.update_status()

    def set_end(self):
        if self.video is None:
            return
        self.end_frame = self.current_frame()
        self.update_status()

    def commit(self):
        if self.video is None:
            return
        if self.start_frame is None or self.end_frame is None:
            self.status.setText("Set START and END first.")
            return

        f0, f1 = sorted([self.start_frame, self.end_frame])
        t0 = frame_to_timecode(f0, self.fps)
        t1 = frame_to_timecode(f1, self.fps)

        etype = self.type_box.currentText()

        if etype == "swap":
            a = self.a_box.currentText()
            b = self.b_box.currentText()
            if not a or not b or a == b:
                self.status.setText("Swap requires two different individuals.")
                return
            e = Edit(type="swap", t0_frame=f0, t1_frame=f1, t0=t0, t1=t1, a=a, b=b)

        else:
            ind = self.ind_box.currentText()
            if not ind:
                self.status.setText("Select an individual to blank.")
                return
            e = Edit(type="blank", t0_frame=f0, t1_frame=f1, t0=t0, t1=t1, ind=ind)

        self.edits.append(e)
        self.start_frame = None
        self.end_frame = None
        self.update_status()

    def undo(self):
        if self.edits:
            self.edits.pop()
        self.update_status()

    def save_json(self):
        if not self.edits:
            self.status.setText("No edits to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save edits JSON", "edits.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.edits], f, indent=2)
        self.status.setText(f"Saved {len(self.edits)} edits -> {os.path.basename(path)}")

    def update_status(self):
        if self.video is None:
            self.status.setText("No video loaded.")
            return

        cf = self.current_frame()

        try:
            n_frames = len(self.video)
        except Exception:
            n_frames = "?"

        msg = f"Frame {cf}/{n_frames} | t={frame_to_timecode(cf, self.fps)} | "

        msg += f"START={self.start_frame if self.start_frame is not None else '-'} "
        msg += f"END={self.end_frame if self.end_frame is not None else '-'} "

        msg += f"| edits={len(self.edits)}"

        if self.individuals:
            msg += f" | inds={len(self.individuals)}"

        self.status.setText(msg)

def main():
    viewer = napari.Viewer()
    widget = AnnotatorWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()


if __name__ == "__main__":
    main()