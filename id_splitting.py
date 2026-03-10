from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def split_trajectories(
    input_h5_path,
    output_h5_path,
    min_jump_distance=200,
    min_gap_size=15,
    use_jump=True,
    use_gap=True,
    individual_key="bird",
    start_index=1,
    hdf_key="df_with_missing",
    split_bodypart=None,
    skip_individuals=None,
):
    """
    Load a DLC h5, split recycled IDs based on jump distance, NaN gaps, or both,
    rename individuals, rebuild DLC-style wide format, and save the new h5.

    Important:
    - If use_gap=True and use_jump=True, splitting uses AND logic:
      split only when a valid point reappears after a long gap AND that point is
      far enough from the last valid point before the gap.
    - Missing frames are restored by reindexing to the full frame axis, because
      pandas.stack() drops NaN rows.

    Returns
    -------
    split_df : pd.DataFrame
        Long-format split dataframe.
    mapping : pd.DataFrame
        Mapping from original individual + segment to new individual name.
    wide : pd.DataFrame
        Rebuilt DLC-style wide dataframe.
    """
    if skip_individuals is None:
        skip_individuals = []

    if not use_jump and not use_gap:
        raise ValueError("At least one of use_jump or use_gap must be True.")

    raw = pd.read_hdf(input_h5_path)

    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Input file does not contain DLC MultiIndex columns.")

    raw.columns = raw.columns.set_names(
        ["scorer", "individuals", "bodyparts", "coords"]
    )
    scorer = raw.columns.get_level_values("scorer")[0]

    # Full frame axis from original DLC dataframe index
    full_time = pd.Index(raw.index.to_numpy(), name="time")

    # Long format; note stack() drops NaNs, so later we reindex back to full_time
    df_long = (
        raw.stack(["scorer", "individuals", "bodyparts", "coords"])
        .rename("value")
        .reset_index()
        .rename(columns={"level_0": "time", "index": "time"})
    )

    split_groups = []

    for ind, group in df_long.groupby("individuals", sort=False):
        group = group.copy()

        if ind in skip_individuals:
            group["segment"] = 0
            split_groups.append(group)
            continue

        xy = group[group["coords"].isin(["x", "y"])].copy()

        if split_bodypart is not None:
            xy = xy[xy["bodyparts"] == split_bodypart].copy()

        if xy.empty:
            group["segment"] = 0
            split_groups.append(group)
            continue

        pivoted = xy.pivot(
            index="time",
            columns=["bodyparts", "coords"],
            values="value",
        ).dropna(axis=1, how="all")

        if pivoted.empty:
            group["segment"] = 0
            split_groups.append(group)
            continue

        if split_bodypart is not None:
            coords_present = pivoted.columns.get_level_values("coords")
            if ("x" not in coords_present) or ("y" not in coords_present):
                group["segment"] = 0
                split_groups.append(group)
                continue

            x = pivoted.xs("x", level="coords", axis=1).iloc[:, 0]
            y = pivoted.xs("y", level="coords", axis=1).iloc[:, 0]
        else:
            x = pivoted.xs("x", level="coords", axis=1).mean(axis=1)
            y = pivoted.xs("y", level="coords", axis=1).mean(axis=1)

        # CRITICAL FIX: restore missing frames so gap detection works
        x = x.reindex(full_time)
        y = y.reindex(full_time)

        valid = x.notna() & y.notna()

        # Previous valid coordinate and previous valid frame
        prev_x = x.where(valid).ffill().shift(1)
        prev_y = y.where(valid).ffill().shift(1)

        prev_valid_time = pd.Series(
            np.where(valid, full_time.to_numpy(), np.nan),
            index=full_time,
        ).ffill().shift(1)

        gap_size = pd.Series(full_time.to_numpy(), index=full_time) - prev_valid_time

        jump = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

        # A "big gap" is evaluated at the first valid frame after missing data
        has_big_gap = valid & gap_size.ge(min_gap_size).fillna(False)

        # A "big jump" is evaluated at valid frames against previous valid point
        has_big_jump = valid & jump.gt(min_jump_distance).fillna(False)

        # Split logic
        if use_gap and use_jump:
            # AND logic
            split_event = has_big_gap & has_big_jump
        elif use_gap:
            # gap-only
            split_event = has_big_gap
        else:
            # jump-only
            split_event = has_big_jump

        segment = split_event.cumsum().astype(int)

        # Map only original observed rows back to their segment
        group["segment"] = group["time"].map(segment).fillna(0).astype(int)
        split_groups.append(group)

    split_df = pd.concat(split_groups, ignore_index=True)
    split_df["individuals_original"] = split_df["individuals"].astype(str)

    mapping = (
        split_df[["individuals_original", "segment"]]
        .drop_duplicates()
        .sort_values(["individuals_original", "segment"])
        .reset_index(drop=True)
    )

    mapping["individuals_new"] = [
        f"{individual_key}{i}"
        for i in range(start_index, start_index + len(mapping))
    ]

    split_df = split_df.merge(
        mapping,
        on=["individuals_original", "segment"],
        how="left",
    )

    split_df["individuals"] = split_df["individuals_new"]
    split_df = split_df.drop(columns=["individuals_new"])

    wide = split_df.pivot_table(
        index="time",
        columns=["individuals", "bodyparts", "coords"],
        values="value",
        aggfunc="first",
    ).sort_index(axis=1)

    wide.columns = pd.MultiIndex.from_tuples(
        [(scorer, ind, bp, coord) for ind, bp, coord in wide.columns],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    output_h5_path = Path(output_h5_path)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_hdf(output_h5_path, key=hdf_key, mode="w")

    return split_df, mapping, wide

def plot_trajectories(
        df,
        bodypart="middle",
        individuals=None,
        time_start=None,
        time_stop=None,
        frame_width=None,
        frame_height=None,
        invert_yaxis=True,
        label_last_point=True,
        title="Trajectories",
    ):
    """
    Plot selected individuals together using one bodypart.

    Args:
        df : pd.DataFrame
            DLC-style wide dataframe.
        bodypart : str, default='middle'
            Bodypart to plot.
        individuals : list[str] or None
            Individuals to plot. None = all.
        time_start, time_stop : int or None
            Optional time limits.
        frame_width, frame_height : int/float or None
            Optional axis limits.
        invert_yaxis : bool, default=True
            Reverse y-axis for image coordinates.
        label_last_point : bool, default=True
            Label last point of each trajectory.
    """
    df = df.copy()
    df.columns = df.columns.set_names(["scorer", "individuals", "bodyparts", "coords"])

    scorer = df.columns.get_level_values("scorer")[0]
    available = df.columns.get_level_values("individuals").unique().tolist()

    if individuals is None:
        individuals = available

    fig = go.Figure()
    for ind in individuals:
        if ind not in available:
            continue
        try:
            x = df[(scorer, ind, bodypart, "x")]
            y = df[(scorer, ind, bodypart, "y")]
        except KeyError:
            continue

        sub = pd.DataFrame(
            {"time": df.index, "x": x.to_numpy(), "y": y.to_numpy()}
        ).dropna(subset=["x", "y"])

        if time_start is not None:
            sub = sub[sub["time"] >= time_start]
        if time_stop is not None:
            sub = sub[sub["time"] <= time_stop]

        if sub.empty:
            continue

        fig.add_trace(
            go.Scattergl(
                x=sub["x"],
                y=sub["y"],
                mode="lines",
                name=ind,
                customdata=sub["time"],
                hovertemplate=f"{ind}<br>time=%{{customdata}}<br>x=%{{x}}<br>y=%{{y}}<extra></extra>",
            )
        )

        if label_last_point:
            last = sub.iloc[-1]
            fig.add_trace(
                go.Scattergl(
                    x=[last["x"]],
                    y=[last["y"]],
                    mode="markers+text",
                    text=[ind],
                    textposition="top center",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=f"{title} ({bodypart})",
        template="plotly_white",
        xaxis_title="x",
        yaxis_title="y",
        legend_title="individual",
        width=900,
        height=700,
    )

    if frame_width is not None:
        fig.update_xaxes(range=[0, frame_width])

    if frame_height is not None:
        if invert_yaxis:
            fig.update_yaxes(range=[frame_height, 0])
        else:
            fig.update_yaxes(range=[0, frame_height])

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig