# %%

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from movement.filtering import filter_by_confidence, interpolate_over_time
from movement.io import load_poses, save_poses
from movement.kinematics import compute_pairwise_distances
from movement.utils.reports import report_nan_values
from movement.utils.vector import compute_norm

from scipy.spatial.transform import Rotation as R

# Hide attributes globally
xr.set_options(display_expand_attrs=False)
# %%
# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget

# %%
# Input data paths
notebook_path = Path(glob.glob("notebook_seabirds.ipynb")[0]).resolve()
data_dir = notebook_path.parent / "data"
filepath = (
    data_dir
    / "second-iter"
    / "FILE00009_sDLC_DekrW32_seabirdNov6shuffle1_snapshot_170_el_filtered.h5"
)
output_dir = notebook_path.parent  / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# Vessel size: 8.55 x 2.95 m
boat_max_length_in_m = 8.55  # m
boat_max_width_in_m = 2.95  # m

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper functions


def get_data_for_load_from_numpy(df):
    """Get array from dataframe to use "from numpy" function"""
    list_individuals = sorted(df.columns.get_level_values("individuals").unique())
    list_keypoints = sorted(df.columns.get_level_values("bodyparts").unique())
    n_keypoints = len(list_keypoints)
    n_individuals = len(list_individuals)

    # position array
    df_position = df.drop(columns=[col for col in df.columns if "likelihood" in col])

    # get number of frames
    position_array = df_position.to_numpy()
    position_array = position_array.reshape(
        df.shape[0],
        2,
        n_keypoints,
        n_individuals,
        order="F",
    )

    # confidence array
    df_confidence = df.drop(
        columns=[col for col in df.columns if "likelihood" not in col]
    )
    confidence_array = df_confidence.to_numpy()
    confidence_array = confidence_array.reshape(
        df.shape[0],
        n_keypoints,
        n_individuals,
        order="F",
    )

    return position_array, confidence_array, list_individuals, list_keypoints


def compute_rotation_to_align_y_axis(vec):
    """Compute rotation to align y-axis"""
    rrot, _rssd = R.align_vectors(
        np.array([[0, 1, 0]]),  # Vector components observed in initial frame A
        vec,  # Vector components observed in another frame B
        return_sensitivity=False,
    )

    return rrot


def add_z_coord_to_position_array(position_array):
    """Add z coordinate to position array"""
    return xr.concat(
        [
            position_array,
            xr.full_like(
                position_array.sel(space="x"),
                0,
            ).expand_dims(space=["z"]),
        ],
        dim="space",
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read input data as pandas dataframe
df = pd.read_hdf(filepath)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get dataset with bird data only
if (filepath.parent / (filepath.stem + "_birds.h5")).exists():
    ds_birds = load_poses.from_dlc_file(filepath.parent / (filepath.stem + "_birds.h5"))
else:
    columns_to_drop = [col for col in df.columns if "single" in col]
    df_birds = df.drop(columns=columns_to_drop)

    position_array, confidence_array, list_individuals, list_keypoints = (
        get_data_for_load_from_numpy(df_birds)
    )

    ds_birds = load_poses.from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=list_individuals,
        keypoint_names=list_keypoints,
        # fps=30,
    )

    # export to file importable in napari
    # To visualise exported file, follow this guide:
    # https://movement.neuroinformatics.dev/user_guide/gui.html
    save_poses.to_dlc_file(ds_birds, filepath.parent / (filepath.stem + "_birds.h5"))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get dataset with boat data only
if (filepath.parent / (filepath.stem + "_boat.h5")).exists():
    ds_boat = load_poses.from_dlc_file(filepath.parent / (filepath.stem + "_boat.h5"))
else:
    columns_to_drop = [col for col in df.columns if "bird" in col[1]]
    df_boat = df.drop(columns=columns_to_drop)

    position_array, confidence_array, list_individuals, list_keypoints = (
        get_data_for_load_from_numpy(df_boat)
    )

    ds_boat = load_poses.from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=list_individuals,
        keypoint_names=list_keypoints,
        # fps=30,
    )

    # export for importable in napari
    save_poses.to_dlc_file(
        ds_boat, filepath.parent / (filepath.stem + "_boat.h5"), split_individuals=False
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Filter low-confidence values
# (values below the threshold are set to nan)
confidence_threshold = 0.5

boat_position = filter_by_confidence(
    ds_boat.position, ds_boat.confidence, threshold=confidence_threshold
)
birds_position = filter_by_confidence(
    ds_birds.position, ds_birds.confidence, threshold=confidence_threshold
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Linearly interpolate boat points
# (gaps with nan are linearly inteprolated)
boat_position_interp = interpolate_over_time(
    boat_position,
    method="linear",
    print_report=True,
)  # there should be no nans after interp


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute rotation to BCS (boat coordinate system)
# - origin : centroid of all boat keypoints per frame
# - y-axis: vector from boat centroid to tip keypoint
# - x-axis: perpendicular to y-axis, points to left side of the boat
#   (it is a rotation from the image coordinate system (ICS))

# Note: we need to flip the x-coord to match the "classic plot"
# coordinate system (x-axis from left to right, y-axis from bottom to top).
# We cannot rotate the ICS into the "classic plot", it needs a flip of
# the x-axis.


# compute origin
boat_position_3d = add_z_coord_to_position_array(boat_position_interp)
boat_centroid_3d = boat_position_3d.mean("keypoints")

# compute y-axis
boat_y_axis_3d = (
    boat_position_3d.sel(keypoints="boatTip") - boat_centroid_3d
).drop_vars(["keypoints"])
boat_centroid_3d = boat_centroid_3d.drop_vars("individuals").squeeze()
boat_y_axis_3d = boat_y_axis_3d.drop_vars("individuals").squeeze()

# compute rotation from ICS y-axis to BCS y-axis
rotation2boat = xr.apply_ufunc(
    lambda v: compute_rotation_to_align_y_axis(v),
    boat_y_axis_3d,
    input_core_dims=[["space"]],
    vectorize=True,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute bird keypoints in BCS (translated and rotated)
birds_position_3d = add_z_coord_to_position_array(birds_position)

birds_position_3d_BCS = xr.apply_ufunc(
    lambda rot, trans, vec: rot.apply(vec - trans),
    rotation2boat,  # rot
    boat_centroid_3d,  # trans
    birds_position_3d,  # vec
    input_core_dims=[[], ["space"], ["space"]],
    output_core_dims=[["space"]],
    vectorize=True,
)

# drop z coordinate
birds_position_BCS = birds_position_3d_BCS.drop_sel(space="z")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply same transform to boat points
boat_position_3d_BCS = xr.apply_ufunc(
    lambda rot, trans, vec: rot.apply(vec - trans),
    rotation2boat,  # rot
    boat_centroid_3d,  # trans
    boat_position_3d,  # vec
    input_core_dims=[[], ["space"], ["space"]],
    output_core_dims=[["space"]],
    vectorize=True,
)

# drop z coordinate
boat_position_BCS = boat_position_3d_BCS.drop_sel(space="z")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply scaling

# Compute boat width and length per frame in pixels
boat_width = compute_pairwise_distances(
    boat_position_BCS, dim="keypoints", pairs={"boatBL": "boatBR"}
)
# boat_width.name = "position"

boat_midpoint_BL_BR = boat_position_BCS.sel(keypoints=["boatBL", "boatBR"]).mean(
    dim="keypoints"
)
boat_length = compute_norm(
    boat_position_BCS.sel(keypoints="boatTip") - boat_midpoint_BL_BR
).squeeze()


# check with plot
plt.figure()
boat_width.plot(label="width")
boat_length.plot(label="length")
plt.xlabel("time (frames)")
plt.ylabel("distance (pixels)")
plt.legend()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Express spatial coordinates in meters

# We use boat length to scale the data
scale_factor = (
    boat_max_length_in_m / boat_length
)  # (boat_max_width_in_m / boat_width) - looks nosier
boat_position_BCS_in_m = boat_position_BCS * scale_factor
birds_position_BCS_in_m = birds_position_BCS * scale_factor


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot data in BCS

# Select a time slice for clarity (frames 0 to 654)
time_slice = slice(0, 1400)

fig, ax = plt.subplots(1, 1)

# plot bird data and color by individual
cmap = plt.get_cmap("tab10")
color_array = cmap(np.arange(len(birds_position_BCS_in_m.individuals)))

for i, ind in enumerate(birds_position_BCS_in_m.individuals):
    # bird centroids
    ax.scatter(
        birds_position_BCS_in_m.sel(time=time_slice, individuals=ind, space="x").mean(
            "keypoints"
        ),
        birds_position_BCS_in_m.sel(time=time_slice, individuals=ind, space="y").mean(
            "keypoints"
        ),
        5,
        color=color_array[i],
        label=ind.item(),
    )

ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1))

# plot boat centroid
sc = ax.scatter(
    boat_position_BCS_in_m.sel(time=time_slice, space="x").mean("keypoints"),
    boat_position_BCS_in_m.sel(time=time_slice, space="y").mean("keypoints"),
    10,
    c=np.arange((time_slice.stop - time_slice.start) + 1),
    cmap="plasma",
    marker="*",
)

# plot boat keypoints in time
for boat_keypoint in ["boatTip", "boatBL", "boatBR"]:
    ax.scatter(
        boat_position_BCS_in_m.sel(time=time_slice, keypoints=boat_keypoint, space="x"),
        boat_position_BCS_in_m.sel(time=time_slice, keypoints=boat_keypoint, space="y"),
        10,
        c=np.arange((time_slice.stop - time_slice.start) + 1),
        cmap="plasma",
    )

ax.set_xlabel("x_BCS (m)")
ax.set_ylabel("y_BCS (m)")
ax.set_aspect("equal")

# add colorbar
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("frames")

# put legend top left
ax.legend(loc="upper left")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save movement datasets
birds_position_BCS_in_m.to_netcdf(output_dir / "birds_position_BCS_in_m.nc")
boat_position_BCS_in_m.to_netcdf(output_dir / "boat_position_BCS_in_m.nc")

# %%
# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Split IDs


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
        group = group[group["position"].notna()]

        segments.append(group)

    return pd.concat(segments, ignore_index=True)


def get_significant_gaps(is_valid, min_gap_size):
    """
    Identify where significant gaps (>= min_gap_size consecutive NaNs) occur.
    Returns a Series of segment IDs.
    """
    # Identify consecutive runs of the same value
    # .ne() --> True where a transition occurs
    # .cumsum() ---> runnning ID (Since True = 1 and False = 0, this increments by 1 each time there's a transition.)
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


# Apply it
df_birds_position = birds_position_BCS_in_m.to_dataframe().reset_index()
df_with_segments = add_segment_ids(df_birds_position, min_gap_size=1)
print(df_with_segments)
print(df_with_segments.duplicated().sum())

# %%%%%%
# Convert to xarray dataarray

# overwrite colum
df_with_segments["new_individuals"] = df_with_segments["individuals"].str[
    :-1
] + df_with_segments["segment"].astype(str)


print(df_with_segments)
print(df_with_segments.duplicated().sum())
# %%
df_with_segments = df_with_segments.drop(columns=["individuals"])
# --- HERE WE HAVE NOW DUPLICATE ROWS
# print(df_with_segments.duplicated().sum()) returns 48

df_with_segments.rename(columns={"new_individuals": "individuals"})
print(df_with_segments.duplicated().sum())
# %%%%%
# convert to xarray
da_split = (
    df_with_segments.loc[:, ["time", "space", "keypoints", "individuals", "position"]]
    .set_index(["time", "space", "keypoints", "individuals"])["position"]
    .to_xarray()
)

# %%
