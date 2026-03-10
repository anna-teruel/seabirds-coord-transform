import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def create_labeled_video(
    video_path,
    h5_path,
    output_path,
    individuals="all",
    bodyparts=("middle",),
    pcutoff=0.0,
    trailpoints=0,
    draw_lines=False,
    skeleton=None,
    dotsize=4,
    linewidth=2,
    fps=None,
    codec="mp4v",
    show_labels=False,
):
    """
    Create a labeled video from a DLC-style H5 file without relying on DLC's
    create_labeled_video() function.

    Args:
        video_path : str
            Input video path.
        h5_path : str
            DLC-style H5 path.
        output_path : str
            Output video path.
        individuals : "all" or list
            Individuals to draw.
        bodyparts : tuple/list or "all"
            Bodyparts to draw.
        pcutoff : float
            Minimum likelihood threshold.
        trailpoints : int
            Number of previous frames to show as trail.
        draw_lines : bool
            Whether to draw skeleton lines.
        skeleton : list of tuple
            Example: [("leftwing", "middle"), ("middle", "rightwing")]
        dotsize : int
            Circle radius.
        linewidth : int
            Line width.
        fps : float or None
            Output fps. If None, uses input video fps.
        codec : str
            FourCC codec.
        show_labels : bool
            Whether to write individual names next to points.
        """
    df = pd.read_hdf(h5_path)
    df.columns = df.columns.set_names(["scorer", "individuals", "bodyparts", "coords"])

    scorer = df.columns.get_level_values("scorer")[0]
    all_inds = df.columns.get_level_values("individuals").unique().tolist()
    all_bps = df.columns.get_level_values("bodyparts").unique().tolist()

    if individuals == "all":
        individuals = all_inds
    if bodyparts == "all":
        bodyparts = all_bps

    if skeleton is None:
        skeleton = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None:
        fps = input_fps

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open output video: {output_path}")

    # stable per-individual colors
    cmap = {}
    for i, ind in enumerate(individuals):
        rng = np.random.default_rng(i + 1)
        cmap[ind] = tuple(int(v) for v in rng.integers(50, 255, size=3))

    frame_idx = 0
    n_frames_h5 = len(df)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in tqdm(range(total_frames), desc="Rendering video"):
        ret, frame = cap.read()
        if not ret:
            break

        row = df.iloc[frame_idx]

        # Draw trails and points
        for ind in individuals:
            color = cmap[ind]

            # bodypart points
            for bp in bodyparts:
                try:
                    x = row[(scorer, ind, bp, "x")]
                    y = row[(scorer, ind, bp, "y")]
                except KeyError:
                    continue

                try:
                    lik = row[(scorer, ind, bp, "likelihood")]
                except KeyError:
                    lik = 1.0

                if pd.isna(x) or pd.isna(y) or pd.isna(lik) or lik < pcutoff:
                    continue

                x_i = int(round(float(x)))
                y_i = int(round(float(y)))

                if 0 <= x_i < width and 0 <= y_i < height:
                    cv2.circle(frame, (x_i, y_i), dotsize, color, -1)
                    if show_labels:
                        cv2.putText(
                            frame,
                            f"{ind}:{bp}",
                            (x_i + 5, y_i - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

                # trail
                if trailpoints > 0:
                    start = max(0, frame_idx - trailpoints)
                    pts = []
                    for t in range(start, frame_idx + 1):
                        try:
                            xt = df.iloc[t][(scorer, ind, bp, "x")]
                            yt = df.iloc[t][(scorer, ind, bp, "y")]
                            lt = df.iloc[t].get((scorer, ind, bp, "likelihood"), 1.0)
                        except Exception:
                            continue

                        if pd.isna(xt) or pd.isna(yt) or pd.isna(lt) or lt < pcutoff:
                            continue

                        xt_i = int(round(float(xt)))
                        yt_i = int(round(float(yt)))
                        if 0 <= xt_i < width and 0 <= yt_i < height:
                            pts.append((xt_i, yt_i))

                    if len(pts) >= 2:
                        for p1, p2 in zip(pts[:-1], pts[1:]):
                            cv2.line(frame, p1, p2, color, linewidth)

            # skeleton
            if draw_lines and skeleton:
                for bp1, bp2 in skeleton:
                    try:
                        x1 = row[(scorer, ind, bp1, "x")]
                        y1 = row[(scorer, ind, bp1, "y")]
                        l1 = row.get((scorer, ind, bp1, "likelihood"), 1.0)

                        x2 = row[(scorer, ind, bp2, "x")]
                        y2 = row[(scorer, ind, bp2, "y")]
                        l2 = row.get((scorer, ind, bp2, "likelihood"), 1.0)
                    except KeyError:
                        continue

                    if any(pd.isna(v) for v in [x1, y1, x2, y2, l1, l2]):
                        continue
                    if l1 < pcutoff or l2 < pcutoff:
                        continue

                    p1 = (int(round(float(x1))), int(round(float(y1))))
                    p2 = (int(round(float(x2))), int(round(float(y2))))
                    if (
                        0 <= p1[0] < width and 0 <= p1[1] < height
                        and 0 <= p2[0] < width and 0 <= p2[1] < height
                    ):
                        cv2.line(frame, p1, p2, color, linewidth)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved labeled video to: {output_path}")