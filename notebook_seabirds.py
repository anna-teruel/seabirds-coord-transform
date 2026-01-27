# %%

import pandas as pd
from movement.io import load_poses, save_poses
from pathlib import Path
import xarray as xr
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import glob

# %%
# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget

# %%
# Input data paths
notebook_path = Path(glob.glob("notebook_seabirds.ipynb")[0]).resolve()
data_dir = notebook_path.parent / "data" / "second-iter"
filepath = (
    data_dir / "FILE00009_sDLC_DekrW32_seabirdNov6shuffle1_snapshot_170_el_filtered.h5"
)


# %%
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


# %%
# Read input data as pandas dataframe
df = pd.read_hdf(filepath)


# %%
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

# %%
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

# %%
# Express coordinates in BCS (boat coordinate system)
# - origin : centroid of all boat keypoints per frame
# - y-axis: vector from boat centroid to tip keypoint
# - x-axis: perpendicular to y-axis, points to left side of the boat
#   (it is a rotation from the image coordinate system (ICS))

# Note: we need to flip the x-coord to match the "classic plot"
# coordinate system (x-axis from left to right, y-axis from bottom to top).
# We cannot rotate the ICS into the "classic plot", it needs a flip of
# the x-axis.

# compute origin
boat_position = ds_boat.position
boat_position_3d = add_z_coord_to_position_array(ds_boat.position)

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

# rotation2boat = rotation2boat.drop_vars("individuals").squeeze()

# %%
# Compute bird keypoints in BCS (translated and rotated)
birds_position_3d = add_z_coord_to_position_array(ds_birds.position)

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

# %%
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

# %%
# Plot data in BCS

# Select a time slice for clarity (frames 0 to 654)
time_slice = slice(0, 1999)

fig, ax = plt.subplots(1, 1)

# plot bird data and color by individual
cmap = plt.get_cmap("tab10")
color_array = cmap(np.arange(len(birds_position_BCS.individuals)))

for i, ind in enumerate(birds_position_BCS.individuals):
    # bird centroids
    ax.scatter(
        birds_position_BCS.sel(time=time_slice, individuals=ind, space="x").mean(
            "keypoints"
        ),
        birds_position_BCS.sel(time=time_slice, individuals=ind, space="y").mean(
            "keypoints"
        ),
        5,
        color=color_array[i],
        label=ind.item(),
    )

ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1))

# plot boat centroid
sc = ax.scatter(
    boat_position_BCS.sel(time=time_slice, space="x").mean("keypoints"),
    boat_position_BCS.sel(time=time_slice, space="y").mean("keypoints"),
    10,
    c=np.arange((time_slice.stop - time_slice.start) + 1),
    cmap="plasma",
    marker="*",
)

# plot boat keypoints in time
for boat_keypoint in ["boatTip", "boatBL", "boatBR"]:
    ax.scatter(
        boat_position_BCS.sel(time=time_slice, keypoints=boat_keypoint, space="x"),
        boat_position_BCS.sel(time=time_slice, keypoints=boat_keypoint, space="y"),
        10,
        c=np.arange((time_slice.stop - time_slice.start) + 1),
        cmap="plasma",
    )

ax.set_xlabel("x_BCS (pixels)")
ax.set_ylabel("y_BCS (pixels)")
ax.set_aspect("equal")

# add colorbar
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("frames")

# put legend top left
ax.legend(loc="upper left")
# %%
