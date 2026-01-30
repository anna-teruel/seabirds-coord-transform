"""A notebook to postprocess DLC trajectories in BCS.

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
from movement.kinematics import compute_forward_displacement

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

fps = 30  # frames per second (video)
min_gap_size = 15  # in frames, for splitting IDs
min_n_frames_with_data = fps * 15  # per ID, for filtering out short trajectories

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
    for (individual), group in df.groupby(["individuals"]):
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

# convert to dataframe first
df_birds_position = birds_position_BCS_in_m.to_dataframe().reset_index()
df_with_segments = add_segment_ids(df_birds_position, min_gap_size=min_gap_size)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Convert back to xarray dataarray

# define new ID based on "segment"
df_with_segments["new_individuals"] = df_with_segments["individuals"].str[
    :-1
] + df_with_segments["segment"].astype(str).str.zfill(3)
df_with_segments = df_with_segments.drop(columns=["individuals"])
df_with_segments = df_with_segments.rename(columns={"new_individuals": "individuals"})

# convert to xarray data array
birds_position_BCS_m_split = (
    df_with_segments.loc[:, ["time", "space", "keypoints", "individuals", "position"]]
    .set_index(["time", "space", "keypoints", "individuals"])["position"]
    .to_xarray()
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Filter out short trajectories

# Compute number of frames with at least one keypoint per id
valid_frames_per_id = (
    birds_position_BCS_m_split.notnull()
    .all(dim="space").any(dim="keypoints").sum(dim="time")
)

# filter
birds_position_BCS_m_split = birds_position_BCS_m_split.sel(
    individuals=valid_frames_per_id >= min_n_frames_with_data
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Remove datapoints with a big jump

# compute_forward_displacement(birds_position_BCS_m_split)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot data
# Select a time slice for clarity (frames 0 to 654)
time_slice = slice(0, 3000)

fig, ax = plt.subplots(1, 1)

# plot bird data and color by individual
cmap = plt.get_cmap("tab20")
n_individuals = len(birds_position_BCS_m_split.individuals)
color_array = cmap(np.arange(n_individuals) % cmap.N)

for i, ind in enumerate(birds_position_BCS_m_split.individuals):
    # Get the data for this individual
    x_data = birds_position_BCS_m_split.sel(time=time_slice, individuals=ind, space="x").mean("keypoints")
    y_data = birds_position_BCS_m_split.sel(time=time_slice, individuals=ind, space="y").mean("keypoints")
    
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

# add colorbar
# cbar = fig.colorbar(sc, ax=ax)
# cbar.set_label("frames")

# put legend top left
# ax.legend(loc="upper left")

# %%
# Plot an individual bird over time
plt.figure()
plot_centroid_trajectory(
    birds_position_BCS_m_split.sel(time=time_slice), individual="bird022"
)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
# %%
%matplotlib widget

# %%
