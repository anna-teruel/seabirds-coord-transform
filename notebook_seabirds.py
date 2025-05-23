# %%

import pandas as pd
from movement.io import load_poses, save_poses
from pathlib import Path
import xarray as xr
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

%matplotlib widget

# TODO next:
# - scale by boat width
# - define ROI?

# %%
data_dir = Path("/Users/sofia/swc/project_seabirds/data")
filepath = data_dir / "scaled_video3DLC_DekrW32_seabirdApr22shuffle1_snapshot_140_el.h5"


# %%
def get_data_for_load_from_numpy(df):
    """Get array from df to load "from numpy" """
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
    rrot, rssd = R.align_vectors(
        np.array([[0, 1, 0]]),  # Vector components observed in initial frame A
        vec,  # Vector components observed in another frame B
        return_sensitivity=False,
    )
    return rrot


def add_z_coord_to_position_array(position_array):
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
# Read as pandas dataframe
if str(filepath).endswith(".h5"):
    df = pd.read_hdf(filepath)
else:
    df = pd.read_csv(filepath)


# %%
# Get dataset for birds only
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

# export for importable in napari
save_poses.to_dlc_file(ds_birds, filepath.parent / (filepath.stem + "_birds.h5"))

# %%
# Get dataset for boat only
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
save_poses.to_dlc_file(ds_boat, filepath.parent / (filepath.stem + "_boat.h5"), split_individuals=False)

# %%
# Express coordinates in BCS (boat coordinate system)
# origin : boat centroid
# y-axis: centroid to tip
# x-axis: points to left side of the boat (bc it is a rotation from ICS)

# Note: we need to flip the x-coord because the ICS will never rotate to be the "classic plot" 
# coordinate system

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


# %%
# compute rotation to y-axis
rotation2boat = xr.apply_ufunc(
    lambda v: compute_rotation_to_align_y_axis(v),
    boat_y_axis_3d,
    input_core_dims=[["space"]],
    vectorize=True,
)

# rotation2boat = rotation2boat.drop_vars("individuals").squeeze()

# %%
# compute keypoints in ECS (translated and rotated)
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

fig, ax = plt.subplots(1, 1)

time_slice = slice(0,654)

cmap = plt.get_cmap("turbo")
color_array = cmap(np.linspace(0, 1, len(birds_position_BCS.individuals)))

for i, ind in enumerate(birds_position_BCS.individuals):
    # birds
    ax.scatter(
        birds_position_BCS.sel(time=time_slice, individuals=ind, space="x").mean("keypoints"),
        birds_position_BCS.sel(time=time_slice, individuals=ind, space="y").mean("keypoints"),
        5,
        color=color_array[i],
    )

# plot boat centroid
sc = ax.scatter(
    boat_position_BCS.sel(time=time_slice, space="x").mean("keypoints"),
    boat_position_BCS.sel(time=time_slice, space="y").mean("keypoints"),
    3,
    c=np.arange(time_slice.stop+1),
    cmap="viridis",
)

# plot boat tip
ax.scatter(
    boat_position_BCS.sel(time=time_slice, keypoints="boatTip", space="x"),
    boat_position_BCS.sel(time=time_slice, keypoints="boatTip", space="y"),
    3,
    c=np.arange(time_slice.stop+1),
    cmap="viridis",
)

# plot boat tip
ax.scatter(
    boat_position_BCS.sel(time=time_slice, keypoints="boatBL", space="x"),
    boat_position_BCS.sel(time=time_slice, keypoints="boatBL", space="y"),
    3,
    c=np.arange(time_slice.stop+1),
    cmap="viridis",
)

# plot boat tip
ax.scatter(
    boat_position_BCS.sel(time=time_slice, keypoints="boatBR", space="x"),
    boat_position_BCS.sel(time=time_slice, keypoints="boatBR", space="y"),
    3,
    c=np.arange(time_slice.stop+1),
    cmap="viridis",
)

# ax.invert_yaxis()
ax.set_xlabel("x_BCS (pixels)")
ax.set_ylabel("y_BCS (pixels)")
ax.set_aspect("equal")
# %%
