"""A notebook to postprocess DLC trajectories expressed in BCS.

Requirements: following installation instructions for `movement`
https://movement.neuroinformatics.dev/latest/user_guide/installation.html

Then run this notebook in that conda environment.

"""

# %%

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


from movement.plots import plot_centroid_trajectory
from movement.utils.vector import compute_norm
from movement.filtering import interpolate_over_time, savgol_filter

# Hide attributes globally
xr.set_options(display_expand_attrs=False)

# %%%%%%%%%%%%%%%%%%%%%%%
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%
# Input data
notebook_path = Path(glob.glob("notebook_seabirds.ipynb")[0]).resolve()
input_dir = notebook_path.parent / "output"

boat_netcdf = "boat_position_BCS_in_m.nc"
birds_netcdf = "birds_position_BCS_in_m.nc"


# Postprocessing parameters
fps = 30  # frames per second (video)
min_gap_size = 15  # in frames, for splitting IDs
min_n_frames_with_data = fps * 1  # per ID, for filtering out short trajectories

# for defining reference smooth trajectory
savgol_window_size = 30  # fps=30
savgol_poly_order = 1
interp_method_reference = "akima"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper functions


def add_segment_ids(df, min_gap_size=1):
    """
    Add segment IDs based on NaN gaps in position data.

    Parameters
    ----------
    df : DataFrame
        The trajectory data
    min_gap_size : int
        Minimum number of consecutive NaN frames to trigger a split.
        - min_gap_size=1: split on any NaN (default, strictest)
        - min_gap_size=5: only split if gap is 5+ frames
        - min_gap_size=10: tolerate gaps up to 9 frames
    """

    segments = []

    segment_id_delta = 0
    for _individual, group in df.groupby(["individuals"]):
        # Pivot to get x and y side by side
        pivoted = group.pivot(
            index="time", columns=["keypoints", "space"], values="position"
        )

        # If any x/y coord of a keypoint is not nan, observation is valid
        is_valid = pivoted.notna().any(axis=1)

        segment_id = get_significant_gaps(is_valid, min_gap_size)

        # Apply global offset to make IDs unique across individuals
        segment_id += segment_id_delta
        segment_id_delta = segment_id.max() + 1

        # Map segment IDs back to original rows
        group = group.copy()
        group["segment"] = group["time"].map(segment_id)

        # Optionally: filter out the NaN rows
        # group = group[group["position"].notna()]

        segments.append(group)

    return pd.concat(segments, ignore_index=True)


def get_significant_gaps(is_valid, min_gap_size):
    """
    Identify where significant gaps (>= min_gap_size consecutive NaNs) occur.
    Returns a Series of segment IDs.
    """
    # Identify consecutive runs of the same value
    # .ne() --> True where a transition occurs
    # .cumsum() ---> runnning ID (Since True = 1 and False = 0,
    # this increments by 1 each time there's a transition.)
    runs = is_valid.ne(is_valid.shift()).cumsum()

    # Get the length of each run
    run_lengths = is_valid.groupby(runs).transform("size")

    # A "significant gap" is an invalid run that's long enough
    is_big_gap = (~is_valid) & (run_lengths >= min_gap_size)

    # Segment ID increments each time we EXIT a significant gap
    # (i.e., when we go from big_gap=True to big_gap=False)
    # restarts after a big gap
    # True only where previous was in a gap AND current is not
    # .cumsum() --> running count of exits
    segment_id = (is_big_gap.shift(fill_value=False) & ~is_big_gap).cumsum()

    return segment_id


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load movement dataset
birds_position_BCS_in_m = xr.load_dataarray(input_dir / birds_netcdf)
boat_position_BCS_in_m = xr.load_dataarray(input_dir / boat_netcdf)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Split IDs if gap between DLC IDs is sufficiently large

# Convert to dataframe first
df_birds_position = birds_position_BCS_in_m.to_dataframe().reset_index()

# Split IDs
df_with_segments = add_segment_ids(df_birds_position, min_gap_size=min_gap_size)

# Redefine ID based on "segment"
df_with_segments["individuals"] = df_with_segments["individuals"].str[
    :-1
] + df_with_segments["segment"].astype(str).str.zfill(3)

# Convert to xarray data array
birds_position_BCS_m_split = (
    df_with_segments.loc[:, ["time", "space", "keypoints", "individuals", "position"]]
    .set_index(["time", "space", "keypoints", "individuals"])["position"]
    .to_xarray()
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot before filtering

# Select a slice of time for clarity if desired
time_slice = slice(0, 9000)

fig, ax = plt.subplots(1, 1)

# plot bird data and color by individual
cmap = plt.get_cmap("tab20")
n_individuals = len(birds_position_BCS_m_split.individuals)
color_array = cmap(np.arange(n_individuals) % cmap.N)

for i, ind in enumerate(birds_position_BCS_m_split.individuals):
    # Get the data for this individual
    x_data = birds_position_BCS_m_split.sel(
        time=time_slice, individuals=ind, space="x"
    ).mean("keypoints")
    y_data = birds_position_BCS_m_split.sel(
        time=time_slice, individuals=ind, space="y"
    ).mean("keypoints")

    # Check if there's any non-NaN data
    has_data = (~np.isnan(x_data)).any() and (~np.isnan(y_data)).any()

    # bird centroids
    ax.scatter(
        x_data,
        y_data,
        5,
        color=color_array[i],
        label=ind.item() if has_data else None,  # Only label if has data
    )

ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=2)
ax.set_xlabel("x_BCS (m)")
ax.set_ylabel("y_BCS (m)")
ax.set_aspect("equal")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Filter out short trajectories

# Compute number of frames with at least one keypoint per id
valid_frames_per_id = (
    birds_position_BCS_m_split.notnull()
    .all(dim="space")
    .any(dim="keypoints")
    .sum(dim="time")
)

# filter
birds_position_BCS_m_split = birds_position_BCS_m_split.sel(
    individuals=valid_frames_per_id >= min_n_frames_with_data
)

# %%%%%
# plot after filtering
# Select a slice of time for clarity if desired
time_slice = slice(0, 9000)

fig, ax = plt.subplots(1, 1)

# plot bird data and color by individual
cmap = plt.get_cmap("tab20")
n_individuals = len(birds_position_BCS_m_split.individuals)
color_array = cmap(np.arange(n_individuals) % cmap.N)

for i, ind in enumerate(birds_position_BCS_m_split.individuals):
    # Get the data for this individual
    x_data = birds_position_BCS_m_split.sel(
        time=time_slice, individuals=ind, space="x"
    ).mean("keypoints")
    y_data = birds_position_BCS_m_split.sel(
        time=time_slice, individuals=ind, space="y"
    ).mean("keypoints")

    # Check if there's any non-NaN data
    has_data = (~np.isnan(x_data)).any() and (~np.isnan(y_data)).any()

    # bird centroids
    ax.scatter(
        x_data,
        y_data,
        5,
        color=color_array[i],
        label=ind.item() if has_data else None,  # Only label if has data
    )

ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=2)
ax.set_xlabel("x_BCS (m)")
ax.set_ylabel("y_BCS (m)")
ax.set_aspect("equal")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute a smoothed reference trajectory to filter out jumps
smoothed_position = savgol_filter(
    birds_position_BCS_m_split, savgol_window_size, polyorder=savgol_poly_order
)
smoothed_position_interp = interpolate_over_time(
    smoothed_position, method=interp_method_reference
)

# if distance between birds_position_BCS_m_split and  smoothed trajectory
# is above threshold, set datapoints to nan
max_distance_to_smoothed = 3  # in m

distance_to_smoothed = compute_norm(
    birds_position_BCS_m_split - smoothed_position_interp
)

birds_position_BCS_m_split_post = birds_position_BCS_m_split.where(
    distance_to_smoothed <= max_distance_to_smoothed
)

# %%%%%%%%%%%%%%%%%%%%%
# Save postprocessed trajectories

birds_position_BCS_m_split_post.to_netcdf(
    input_dir / "birds_position_BCS_m_postprocessed.nc"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot data
# Select a time slice for clarity 
time_slice = slice(0, 3000)

fig, ax = plt.subplots(1, 1)

# plot bird data and color by individual
cmap = plt.get_cmap("tab20")
n_individuals = len(birds_position_BCS_m_split_post.individuals)
color_array = cmap(np.arange(n_individuals) % cmap.N)

for i, ind in enumerate(birds_position_BCS_m_split_post.individuals):
    # Get the data for this individual
    x_data = birds_position_BCS_m_split_post.sel(
        time=time_slice, individuals=ind, space="x"
    ).mean("keypoints")
    y_data = birds_position_BCS_m_split_post.sel(
        time=time_slice, individuals=ind, space="y"
    ).mean("keypoints")

    # Check if there's any non-NaN data
    has_data = (~np.isnan(x_data)).any() and (~np.isnan(y_data)).any()

    # bird centroids
    ax.scatter(
        x_data,
        y_data,
        5,
        color=color_array[i],
        label=ind.item() if has_data else None,  # Only label if has data
    )

ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=2)
ax.set_xlabel("x_BCS (m)")
ax.set_ylabel("y_BCS (m)")
ax.set_aspect("equal")


# %%%%%%%%%%%%%%%%%%%%%%%%%
# Plot an individual bird over time before filtering out jumps and
# with reference smoothed trajectory
fig, ax = plt.subplots()
plot_centroid_trajectory(
    birds_position_BCS_m_split.sel(time=time_slice),
    individual="bird015",
    ax=ax,
    label="pre",
)
plot_centroid_trajectory(
    smoothed_position_interp.sel(time=time_slice),
    individual="bird015",
    c="r",
    ax=ax,
    label="reference",
)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("before removing data with 'jumps'")
ax.legend()


# %%
# Plot after removing jumps
fig, ax = plt.subplots()
plot_centroid_trajectory(
    birds_position_BCS_m_split_post.sel(time=time_slice),
    individual="bird015",
    ax=ax,
)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("before removing data with 'jumps'")
# %%
