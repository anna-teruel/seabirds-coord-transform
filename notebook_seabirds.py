# %%

import pandas as pd
from movement.io import load_poses, save_poses
from pathlib import Path

# %%
data_dir = Path("/Users/sofia/swc/project_seabirds/data")
filepath = data_dir / "scaled_video3DLC_DekrW32_seabirdApr22shuffle1_snapshot_140_el.h5"
# %%
# Split boat and birds
if str(filepath).endswith(".h5"):
    df = pd.read_hdf(filepath)
else:
    df = pd.read_csv(filepath)


# Get dataframe for boat only
columns_to_drop = [col for col in df.columns if "bird" in col[1]]
df_boat = df.drop(columns=columns_to_drop)


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
    fps=30,
)

# export for importable in napari
save_poses.to_dlc_file(
    ds_birds,
    filepath.parent / (filepath.stem + "_birds.h5")
)

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
    fps=30,
)

# export for importable in napari
save_poses.to_dlc_file(
    ds_boat,
    filepath.parent / (filepath.stem + "_boat.h5")
)

# %%
