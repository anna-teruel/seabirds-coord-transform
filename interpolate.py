import os
import pandas as pd
import numpy as np
import glob
import re
import subprocess
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path


class DataLoader:
    def __init__(self, minutes=None, fps=None):
        """
        Initialize a DataLoader object.

        Args:
            minutes (int, optional): The duration of the video in minutes. Defaults to None.
            fps (int, optional): The frames per second (fps) of the video. Defaults to None.
        """
        self.minutes = minutes
        self.fps = fps

    def read_data(self, input_path):
        """
        Read data from either a single file or a directory.

        Args:
            input_path (str): The path to the file or directory containing the data.

        Raises:
            ValueError: If the provided path does not exist.

        Returns:
            pandas.DataFrame or dict: A DataFrame if a single file is read, or a dictionary of
                                      DataFrames if multiple files are read.
        """
        if os.path.isfile(input_path):  # Check if it's a file
            return self.read_file(input_path)

        elif os.path.isdir(input_path):  # Check if it's a directory
            return self.read_directory(input_path)

        else:
            raise ValueError("Provided path does not exist.")

    def read_file(self, file_path):
        """
        Read data from a single file.

        Args:
            file_path (str): The path to the .h5 file.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the file.
        """
        df = pd.read_hdf(file_path)
        return df

    def read_directory(self, dir_path, suffix=("filtered.h5",)):
        """
        Reads data from all files in a directory that end with a specified suffix.

        Args:
            directory_path (str): The path to the directory containing the .h5 files.
            suffix (tuple, optional): The suffix that the files should end with. Defaults to ('filtered.h5',).

        Returns:
            dict: A dictionary where keys are file names and values are DataFrames containing the data from each file.
        """
        data_dict = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(suffix):
                file_path = os.path.join(dir_path, filename)
                data_dict[filename] = self.read_file(file_path)
        return data_dict
    
    def get_file_name(self, file_path): 
        """_summary_

        Args:
            file_path (str): The path to the .h5 file.

        Returns:
            _type_: _description_
        """        
        file_name_long = Path(file_path).name  # complete file name with extension
        dlc_index = file_name_long.find(
            "DLC"
        )  # all deeplabcut files include project info starts with "DLC"
        file_name = (
            file_name_long[:dlc_index] if dlc_index != -1 else Path(file_path).stem
        )  # remove extra info from title
        return file_name

class Interpolation:
    def __init__(self, threshold=0.90, interpolation_method="linear"):
        self.threshold = threshold
        self.interpolation_method = interpolation_method

    def _interp_triplet(self, df, x_col, y_col, lk_col, label):
        """Interpolate one (x,y,likelihood) triplet in-place."""
        mask = df.loc[:, lk_col] < self.threshold
        n_bad = int(mask.sum())
        if n_bad == 0:
            return

        print(f"Interpolating {n_bad} points for {label}.")

        df.loc[mask, x_col] = np.nan
        df.loc[mask, y_col] = np.nan
        df.loc[mask, lk_col] = np.nan

        df.loc[:, x_col] = df.loc[:, x_col].interpolate(method=self.interpolation_method)
        df.loc[:, y_col] = df.loc[:, y_col].interpolate(method=self.interpolation_method)
        df.loc[:, lk_col] = df.loc[:, lk_col].interpolate(method=self.interpolation_method)

        print(f"NaNs after interpolation for {label} x: {df.loc[:, x_col].isna().sum()}")
        print(f"NaNs after interpolation for {label} y: {df.loc[:, y_col].isna().sum()}")

    def get_interpolation(
        self,
        df,
        bodyparts,
        individuals=None,
        include_single=False,
        single_bodyparts=None,
        verbose_missing=False,
    ):
        """
        Interpolate DLC coordinates.

        Parameters
        ----------
        df : pd.DataFrame
            DLC dataframe with MultiIndex columns.
        bodyparts : list[str]
            Bodyparts to interpolate for multi-animal tracks.
        individuals : list[str] or None
            Which individuals to interpolate. If None, will auto-detect and skip "single".
        include_single : bool
            If True, also interpolate the "single-animal style" columns (no individuals level),
            OR the individual named "single" (only if it has matching columns).
        single_bodyparts : list[str] or None
            Bodyparts to interpolate in the single-animal style block. Defaults to `bodyparts`.
        verbose_missing : bool
            If True, prints missing columns that are skipped.
        """
        if not isinstance(df.columns, pd.MultiIndex):
            raise TypeError("Expected df.columns to be a pandas MultiIndex (DLC-style).")

        scorer = df.columns.get_level_values("scorer")[0]
        colnames = list(df.columns.names)
        has_individuals = "individuals" in colnames

        def _has(col_tuple):
            return col_tuple in df.columns

        if has_individuals:
            all_inds = list(df.columns.get_level_values("individuals").unique())

            if individuals is None:
                individuals_to_use = [i for i in all_inds if i != "single"]
            else:
                individuals_to_use = list(individuals)

            for ind in individuals_to_use:
                for bp in bodyparts:
                    lk_col = (scorer, ind, bp, "likelihood")
                    x_col  = (scorer, ind, bp, "x")
                    y_col  = (scorer, ind, bp, "y")

                    if not (_has(lk_col) and _has(x_col) and _has(y_col)):
                        if verbose_missing:
                            print(f"Skipping {ind}-{bp}: missing columns")
                        continue

                    self._interp_triplet(df, x_col, y_col, lk_col, label=f"{ind} - {bp}")
        return df

    def interpolate_data(self, 
                         input_data, 
                         bodyparts, 
                         individuals=None, 
                         include_single=False, 
                         single_bodyparts=None):
        loader = DataLoader()
        if os.path.isfile(input_data):
            df = loader.read_data(input_data)
            return self.get_interpolation(
                df,
                bodyparts,
                individuals=individuals,
                include_single=include_single,
                single_bodyparts=single_bodyparts,
            )
        elif os.path.isdir(input_data):
            data_dict = loader.read_directory(input_data)
            out = {}
            for key, df in data_dict.items():
                out[key] = self.get_interpolation(
                    df,
                    bodyparts,
                    individuals=individuals,
                    include_single=include_single,
                    single_bodyparts=single_bodyparts,
                )
            return out
        else:
            raise ValueError("Provided path does not exist.")
