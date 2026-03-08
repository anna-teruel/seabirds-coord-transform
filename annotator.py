import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.h5_path = None
        self.dlc_df = None
        self.fps = 30.0
        self.individuals: List[str] = []
        self.bodyparts: List[str] = []
        self.scorer: Optional[str] = None
        self.edits: List[Edit] = []

        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None

        # current-frame editing state
        self.current_frame_idx: Optional[int] = None
        self.current_point_layers: Dict[str, object] = {}
        self._is_refreshing_points = False

        self.trajectory_layer = None
        self.traj_tail_length = 1000

        # ---- UI ----
        layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        self.btn_load_video = QPushButton("Load video (.mp4)")
        self.btn_load_h5 = QPushButton("Load DLC .h5")
        btn_row.addWidget(self.btn_load_video)
        btn_row.addWidget(self.btn_load_h5)
        layout.addLayout(btn_row)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_box = QSpinBox()
        self.fps_box.setRange(1, 240)
        self.fps_box.setValue(30)
        fps_row.addWidget(self.fps_box)
        layout.addLayout(fps_row)

        layout.addWidget(QLabel("Edit type"))
        self.type_box = QComboBox()
        self.type_box.addItems(["swap", "blank"])
        layout.addWidget(self.type_box)

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
        self.btn_save_json = QPushButton("Save JSON")
        self.btn_save_h5 = QPushButton("Save H5")
        save_row.addWidget(self.btn_save_json)
        save_row.addWidget(self.btn_save_h5)
        layout.addLayout(save_row)

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
        self.btn_save_json.clicked.connect(self.save_json)
        self.btn_save_h5.clicked.connect(self.save_corrected_h5)

        # frame-change listener
        self.viewer.dims.events.current_step.connect(self.on_frame_changed)

        self.refresh_visibility()

    # ----------------------------
    # basic UI callbacks
    # ----------------------------
    def on_fps_changed(self, v: int):
        self.fps = float(v)
        self.update_status()

    def refresh_visibility(self):
        etype = self.type_box.currentText()
        is_swap = (etype == "swap")
        self.a_box.setEnabled(is_swap)
        self.b_box.setEnabled(is_swap)
        self.ind_box.setEnabled(not is_swap)

    def current_frame(self) -> int:
        return int(self.viewer.dims.point[0])

    # ----------------------------
    # load video / h5
    # ----------------------------
    def load_video_clicked(self, checked=False):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Video (*.mp4 *.avi *.mov)"
        )
        if not path:
            return

        self.video_path = path
        self.viewer.layers.clear()
        self.video = load_video(self.viewer, path)
        self.viewer.dims.set_point(0, 0)
        self.current_frame_idx = 0
        self.current_point_layers = {}
        self.update_status()

    def _remove_trajectory_layer(self):
        if self.trajectory_layer is not None:
            try:
                if self.trajectory_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.trajectory_layer)
            except Exception:
                pass
        self.trajectory_layer = None


    def draw_all_bodypart_trajectories(self, tail_length=1000):
        if self.dlc_df is None or self.scorer is None:
            return

        self._remove_trajectory_layer()

        tracks_all = []
        track_id = 0

        for ind in self.individuals:
            for bp in self.bodyparts:
                x_col = (self.scorer, ind, bp, "x")
                y_col = (self.scorer, ind, bp, "y")

                if x_col not in self.dlc_df.columns or y_col not in self.dlc_df.columns:
                    continue

                x = self.dlc_df[x_col].to_numpy()
                y = self.dlc_df[y_col].to_numpy()
                frames = np.arange(len(x))

                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() == 0:
                    continue

                arr = np.column_stack([
                    np.full(valid.sum(), track_id),
                    frames[valid],
                    y[valid],
                    x[valid],
                ])
                tracks_all.append(arr)
                track_id += 1

        if not tracks_all:
            return

        tracks_all = np.vstack(tracks_all)

        self.trajectory_layer = self.viewer.add_tracks(
            tracks_all,
            name="trajectories",
            tail_length=tail_length,
            blending="translucent",
        )

    def load_h5_clicked(self, checked=False):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DLC h5", "", "H5 (*.h5 *.hdf5)"
        )
        if not path:
            return

        self._is_refreshing_points = True

        self.h5_path = path
        self.dlc_df = pd.read_hdf(path)

        if not isinstance(self.dlc_df.columns, pd.MultiIndex):
            self.status.setText("File does not contain DLC MultiIndex columns.")
            self._is_refreshing_points = False
            return

        col_names = list(self.dlc_df.columns.names)
        required = {"scorer", "individuals", "bodyparts"}
        if not required.issubset(col_names):
            self.status.setText(f"Unexpected DLC column structure: {col_names}")
            self._is_refreshing_points = False
            return

        self.scorer = self.dlc_df.columns.get_level_values("scorer")[0]
        self.individuals = [str(i) for i in self.dlc_df.columns.get_level_values("individuals").unique()]
        self.bodyparts = [str(bp) for bp in self.dlc_df.columns.get_level_values("bodyparts").unique()]

        self.a_box.clear()
        self.b_box.clear()
        self.ind_box.clear()
        self.a_box.addItems(self.individuals)
        self.b_box.addItems(self.individuals)
        self.ind_box.addItems(self.individuals)
        if len(self.individuals) > 1:
            self.b_box.setCurrentIndex(1)

        self._remove_current_point_layers()
        self._remove_trajectory_layer()

        self.current_frame_idx = self.current_frame()

        # draw trajectories first
        self.draw_all_bodypart_trajectories(tail_length=self.traj_tail_length)

        # then draw editable points
        self.current_point_layers = {}
        self._is_refreshing_points = False
        self._draw_current_frame_points()

        self.update_status()

    # ----------------------------
    # point-layer handling
    # ----------------------------
    def _remove_current_point_layers(self):
        for ind, layer in list(self.current_point_layers.items()):
            try:
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
            except Exception:
                pass
        self.current_point_layers = {}

    def _get_frame_points_for_individual(self, ind, frame_idx):
        pts = []
        labels = []
        frames_meta = []
        inds_meta = []

        for bp in self.bodyparts:
            x_col = (self.scorer, ind, bp, "x")
            y_col = (self.scorer, ind, bp, "y")

            if x_col not in self.dlc_df.columns or y_col not in self.dlc_df.columns:
                continue

            x = self.dlc_df.at[frame_idx, x_col]
            y = self.dlc_df.at[frame_idx, y_col]

            if pd.isna(x) or pd.isna(y):
                continue

            pts.append([frame_idx, y, x])
            labels.append(bp)
            frames_meta.append(frame_idx)
            inds_meta.append(ind)

        if len(pts) == 0:
            return None, None

        points_array = np.asarray(pts, dtype=float)
        features = {
            "bodypart": labels,
            "frame": frames_meta,
            "individual": inds_meta,
        }
        return points_array, features

    def _individual_colors(self):
        cmap = plt.get_cmap("Set2")
        return {ind: [cmap(i % cmap.N)] for i, ind in enumerate(self.individuals)}

    def _draw_current_frame_points(self):
        if self.dlc_df is None or self.scorer is None:
            return

        self._is_refreshing_points = True
        frame_idx = self.current_frame()
        colors = self._individual_colors()

        # first time: create layers once
        if not self.current_point_layers:
            for ind in self.individuals:
                result = self._get_frame_points_for_individual(ind, frame_idx)
                if result[0] is None:
                    continue

                points_array, features = result

                layer = self.viewer.add_points(
                    points_array,
                    name=ind,
                    size=12,
                    face_color=colors[ind],
                    opacity=0.85,
                    features=features,
                )
                layer.text = {
                    "string": "{bodypart}",
                    "size": 8,
                    "color": "white",
                    "translation": np.array([0, -8, 8]),
                }
                self.current_point_layers[ind] = layer

        # later: update existing layers only
        else:
            for ind in self.individuals:
                result = self._get_frame_points_for_individual(ind, frame_idx)

                if ind not in self.current_point_layers:
                    if result[0] is None:
                        continue

                    points_array, features = result
                    layer = self.viewer.add_points(
                        points_array,
                        name=ind,
                        size=12,
                        face_color=colors[ind],
                        opacity=0.85,
                        features=features,
                    )
                    layer.text = {
                        "string": "{bodypart}",
                        "size": 8,
                        "color": "white",
                        "translation": np.array([0, -8, 8]),
                    }
                    self.current_point_layers[ind] = layer
                    continue

                layer = self.current_point_layers[ind]

                if result[0] is None:
                    layer.data = np.empty((0, 3))
                    layer.features = {"bodypart": [], "frame": [], "individual": []}
                else:
                    points_array, features = result
                    layer.data = points_array
                    layer.features = features

        self._is_refreshing_points = False

    def _commit_current_frame_points(self):
        if self.dlc_df is None or self.scorer is None or self._is_refreshing_points:
            return

        frame_idx = self.current_frame_idx
        if frame_idx is None:
            return

        for ind, layer in self.current_point_layers.items():
            if layer.data is None or len(layer.data) == 0:
                continue

            data = np.asarray(layer.data)
            bodyparts = list(layer.features["bodypart"])

            for row, bp in zip(data, bodyparts):
                _, y, x = row

                x_col = (self.scorer, ind, bp, "x")
                y_col = (self.scorer, ind, bp, "y")

                if x_col in self.dlc_df.columns:
                    self.dlc_df.at[frame_idx, x_col] = float(x)
                if y_col in self.dlc_df.columns:
                    self.dlc_df.at[frame_idx, y_col] = float(y)

        # optional: keep likelihood unchanged

    def on_frame_changed(self, event=None):
        if self.dlc_df is None or self._is_refreshing_points:
            return

        new_frame = self.current_frame()

        if self.current_frame_idx is not None and new_frame != self.current_frame_idx:
            self._commit_current_frame_points()

        self.current_frame_idx = new_frame
        self._draw_current_frame_points()
        self.update_status()

    # ----------------------------
    # annotation
    # ----------------------------
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

    # ----------------------------
    # saving
    # ----------------------------
    def save_json(self):
        if not self.edits:
            self.status.setText("No edits to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save edits JSON", "edits.json", "JSON (*.json)"
        )
        if not path:
            return

        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.edits], f, indent=2)

        self.status.setText(f"Saved {len(self.edits)} edits -> {os.path.basename(path)}")

    def save_corrected_h5(self):
        if self.dlc_df is None:
            self.status.setText("No DLC dataframe loaded.")
            return

        # make sure current-frame drags are saved
        self._commit_current_frame_points()

        default_name = "corrected_filtered.h5"
        if self.h5_path is not None:
            stem = os.path.splitext(os.path.basename(self.h5_path))[0]
            default_name = f"{stem}_corrected.h5"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save corrected H5",
            default_name,
            "H5 (*.h5 *.hdf5)"
        )
        if not path:
            return

        self.dlc_df.to_hdf(path, key="df", mode="w")

        if self.edits:
            edits_df = pd.DataFrame([asdict(e) for e in self.edits])
            edits_df.to_hdf(path, key="annotations", mode="a")

        self.status.setText(f"Saved corrected H5 -> {os.path.basename(path)}")

    # ----------------------------
    # status
    # ----------------------------
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

