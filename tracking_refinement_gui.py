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
    type: str
    t0_frame: int
    t1_frame: int
    t0: str
    t1: str
    a: Optional[str] = None
    b: Optional[str] = None
    ind: Optional[str] = None


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

        self.current_frame_idx: Optional[int] = None
        self.current_point_layers: Dict[str, object] = {}
        self._is_refreshing_points = False

        self.trajectory_layer = None
        self.traj_tail_length = 1000

        # add-point mode
        self.add_point_mode = False

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

        layout.addWidget(QLabel("Add missing bodypart"))
        add_row = QHBoxLayout()
        self.add_ind_box = QComboBox()
        self.add_bp_box = QComboBox()
        add_row.addWidget(self.add_ind_box)
        add_row.addWidget(self.add_bp_box)
        layout.addLayout(add_row)

        self.btn_add_point_mode = QPushButton("Add bodypart")
        self.btn_add_point_mode.setCheckable(True)
        layout.addWidget(self.btn_add_point_mode)

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

        self.btn_add_point_mode.toggled.connect(self.toggle_add_point_mode)

        self.viewer.dims.events.current_step.connect(self.on_frame_changed)

        self.refresh_visibility()

    # ----------------------------
    # UI callbacks
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

    def toggle_add_point_mode(self, checked):
        self.add_point_mode = checked
        if checked:
            self.status.setText("Add bodypart mode active: click on the video to place the point.")
        else:
            self.update_status()

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

        # connect mouse callback to video layer
        try:
            video_layer = self.viewer.layers["video"]
            video_layer.mouse_drag_callbacks.append(self.on_viewer_click)
        except Exception:
            pass

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
                    np.full(valid.sum(), track_id),  # track id
                    frames[valid],                   # time
                    y[valid],                        # y
                    x[valid],                        # x
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

        # annotation dropdowns
        self.a_box.clear()
        self.b_box.clear()
        self.ind_box.clear()
        self.a_box.addItems(self.individuals)
        self.b_box.addItems(self.individuals)
        self.ind_box.addItems(self.individuals)
        if len(self.individuals) > 1:
            self.b_box.setCurrentIndex(1)

        # add-point dropdowns
        self.add_ind_box.clear()
        self.add_bp_box.clear()
        self.add_ind_box.addItems(self.individuals)
        self.add_bp_box.addItems(self.bodyparts)

        self._remove_current_point_layers()
        self.current_frame_idx = self.current_frame()

        self._is_refreshing_points = False
        self._draw_current_frame_points()
        self.draw_all_bodypart_trajectories(tail_length=self.traj_tail_length)
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

    def _individual_colors(self):
        cmap = plt.get_cmap("Set2")
        return {ind: [cmap(i % cmap.N)] for i, ind in enumerate(self.individuals)}

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
        features = pd.DataFrame({
            "bodypart": labels,
            "frame": frames_meta,
            "individual": inds_meta,
        })
        return points_array, features

    def _draw_current_frame_points(self):
        if self.dlc_df is None or self.scorer is None:
            return

        self._is_refreshing_points = True
        frame_idx = self.current_frame()
        colors = self._individual_colors()

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

    def on_point_added(self, event):
        layer = event.source

        if self.dlc_df is None or self.scorer is None:
            return

        if len(layer.data) == 0:
            return

        new_index = len(layer.data) - 1
        bp = self.add_bp_box.currentText()

        if not bp:
            self.status.setText("Select a bodypart to add first.")
            return

        # make features a dataframe
        features = pd.DataFrame(layer.features).copy()

        # make sure required columns exist
        for col in ["bodypart", "individual", "frame"]:
            if col not in features.columns:
                features[col] = np.nan

        # if a new point was added, features may be shorter than data
        if len(features) < len(layer.data):
            n_missing = len(layer.data) - len(features)
            extra = pd.DataFrame({
                "bodypart": [np.nan] * n_missing,
                "individual": [np.nan] * n_missing,
                "frame": [np.nan] * n_missing,
            })
            features = pd.concat([features, extra], ignore_index=True)

        features.loc[new_index, "bodypart"] = bp
        features.loc[new_index, "individual"] = layer.name
        features.loc[new_index, "frame"] = self.current_frame()

        layer.features = features
        layer.text = {
            "string": "{bodypart}",
            "size": 8,
            "color": "white",
            "translation": np.array([0, -8, 8]),
        }

        # commit immediately so it is stored in self.dlc_df
        self._commit_current_frame_points()
        self.status.setText(f"Added {bp} to layer {layer.name} at frame {self.current_frame()}.")
    
    def _commit_current_frame_points(self):
        if self.dlc_df is None or self.scorer is None or self._is_refreshing_points:
            return

        frame_idx = self.current_frame_idx
        if frame_idx is None:
            return

        # blank whole frame first
        for ind in self.individuals:
            for bp in self.bodyparts:
                x_col = (self.scorer, ind, bp, "x")
                y_col = (self.scorer, ind, bp, "y")
                lk_col = (self.scorer, ind, bp, "likelihood")

                if x_col in self.dlc_df.columns:
                    self.dlc_df.at[frame_idx, x_col] = np.nan
                if y_col in self.dlc_df.columns:
                    self.dlc_df.at[frame_idx, y_col] = np.nan
                if lk_col in self.dlc_df.columns:
                    self.dlc_df.at[frame_idx, lk_col] = np.nan

        # write back what is currently visible in each point layer
        for ind, layer in self.current_point_layers.items():
            data = np.asarray(layer.data)
            if data.size == 0:
                continue

            features = pd.DataFrame(layer.features)
            if "bodypart" not in features.columns or len(features) != len(data):
                print(f"Skipping commit for {ind}: data/features mismatch")
                continue

            for i in range(len(data)):
                _, y, x = data[i]
                bp = str(features.iloc[i]["bodypart"])

                x_col = (self.scorer, ind, bp, "x")
                y_col = (self.scorer, ind, bp, "y")
                lk_col = (self.scorer, ind, bp, "likelihood")

                if x_col in self.dlc_df.columns:
                    self.dlc_df.at[frame_idx, x_col] = float(x)
                if y_col in self.dlc_df.columns:
                    self.dlc_df.at[frame_idx, y_col] = float(y)
                if lk_col in self.dlc_df.columns:
                    self.dlc_df.at[frame_idx, lk_col] = 1.0


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
    # add missing point
    # ----------------------------
    def on_viewer_click(self, layer, event):
        if not self.add_point_mode:
            return

        if layer is None or layer.name != "video":
            return

        if self.dlc_df is None or self.scorer is None:
            return

        pos = event.position
        if len(pos) < 3:
            return

        frame_idx = int(round(pos[0]))
        y = float(pos[1])
        x = float(pos[2])

        ind = self.add_ind_box.currentText()
        bp = self.add_bp_box.currentText()

        if not ind or not bp:
            self.status.setText("Select individual and bodypart first.")
            return

        x_col = (self.scorer, ind, bp, "x")
        y_col = (self.scorer, ind, bp, "y")
        lk_col = (self.scorer, ind, bp, "likelihood")

        if x_col not in self.dlc_df.columns or y_col not in self.dlc_df.columns:
            self.status.setText(f"Columns not found for {ind} / {bp}.")
            return

        # 1. write directly to dataframe
        self.dlc_df.at[frame_idx, x_col] = x
        self.dlc_df.at[frame_idx, y_col] = y
        if lk_col in self.dlc_df.columns:
            self.dlc_df.at[frame_idx, lk_col] = 1.0

        # 2. update visible layer directly so commit sees it too
        if ind in self.current_point_layers:
            point_layer = self.current_point_layers[ind]
            data = np.asarray(point_layer.data)
            features = pd.DataFrame(point_layer.features).copy()

            # if bodypart already exists in this frame, replace it
            if "bodypart" in features.columns and bp in list(features["bodypart"]):
                idx = list(features["bodypart"]).index(bp)
                data[idx] = [frame_idx, y, x]
            else:
                new_row = np.array([[frame_idx, y, x]], dtype=float)
                if data.size == 0:
                    data = new_row
                else:
                    data = np.vstack([data, new_row])

                new_feat = pd.DataFrame({
                    "bodypart": [bp],
                    "frame": [frame_idx],
                    "individual": [ind],
                })
                features = pd.concat([features, new_feat], ignore_index=True)

            point_layer.data = data
            point_layer.features = features
            point_layer.text = {
                "string": "{bodypart}",
                "size": 8,
                "color": "white",
                "translation": np.array([0, -8, 8]),
            }
        else:
            # fallback: redraw frame if layer doesn't exist yet
            self.current_frame_idx = frame_idx
            self._draw_current_frame_points()

        # 3. commit immediately
        self.current_frame_idx = frame_idx
        self._commit_current_frame_points()

        self.status.setText(f"Added {bp} for {ind} at frame {frame_idx}.")
        pass
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

    def apply_edits_to_dataframe(self):
        if self.dlc_df is None or self.scorer is None:
            return

        for e in self.edits:
            if e.type == "blank":
                ind = e.ind
                if ind is None:
                    continue

                for frame_idx in range(e.t0_frame, e.t1_frame + 1):
                    for bp in self.bodyparts:
                        x_col = (self.scorer, ind, bp, "x")
                        y_col = (self.scorer, ind, bp, "y")
                        lk_col = (self.scorer, ind, bp, "likelihood")

                        if x_col in self.dlc_df.columns:
                            self.dlc_df.at[frame_idx, x_col] = np.nan
                        if y_col in self.dlc_df.columns:
                            self.dlc_df.at[frame_idx, y_col] = np.nan
                        if lk_col in self.dlc_df.columns:
                            self.dlc_df.at[frame_idx, lk_col] = np.nan
            elif e.type == "swap":
                a = e.a
                b = e.b

                if a is None or b is None:
                    continue

                for frame_idx in range(e.t0_frame, e.t1_frame + 1):
                    for bp in self.bodyparts:
                        ax = (self.scorer, a, bp, "x")
                        ay = (self.scorer, a, bp, "y")
                        al = (self.scorer, a, bp, "likelihood")

                        bx = (self.scorer, b, bp, "x")
                        by = (self.scorer, b, bp, "y")
                        bl = (self.scorer, b, bp, "likelihood")

                        if (
                            ax in self.dlc_df.columns and ay in self.dlc_df.columns and
                            bx in self.dlc_df.columns and by in self.dlc_df.columns
                        ):
                            tmp_x = self.dlc_df.at[frame_idx, ax]
                            tmp_y = self.dlc_df.at[frame_idx, ay]
                            tmp_l = self.dlc_df.at[frame_idx, al] if al in self.dlc_df.columns else np.nan

                            self.dlc_df.at[frame_idx, ax] = self.dlc_df.at[frame_idx, bx]
                            self.dlc_df.at[frame_idx, ay] = self.dlc_df.at[frame_idx, by]
                            if al in self.dlc_df.columns and bl in self.dlc_df.columns:
                                self.dlc_df.at[frame_idx, al] = self.dlc_df.at[frame_idx, bl]

                            self.dlc_df.at[frame_idx, bx] = tmp_x
                            self.dlc_df.at[frame_idx, by] = tmp_y
                            if al in self.dlc_df.columns and bl in self.dlc_df.columns:
                                self.dlc_df.at[frame_idx, bl] = tmp_l

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

        # save any current-frame point edits first
        self._commit_current_frame_points()

        # apply blank/swap annotations to dataframe
        self.apply_edits_to_dataframe()

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

        self.dlc_df.to_hdf(
            path,
            key="df_with_missing",
            mode="w",
            format="table"
        )

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

        if self.add_point_mode:
            msg += " | add-point mode ON"

        self.status.setText(msg)
