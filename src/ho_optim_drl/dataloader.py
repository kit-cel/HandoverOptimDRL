"""Data loader for the Vienna dataset"""

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.io

if TYPE_CHECKING:
    from ho_optim_drl.config import Config


def get_filenames(dir_: str, prefix: str) -> list[str]:
    """
    Return the filenames in the specified directory.

    Parameters
    ----------
    dir_ : str
        The directory to search for files.
    prefix : str
        The prefix of the files to search for.

    Returns
    -------
    list[str]
        A list of filenames in the specified directory.
    """
    return [f for f in os.listdir(dir_) if prefix in f]


def load_preprocess_dataset(
    config: "Config",
    data_dir: str,
    rsrp_file: str,
    sinr_file: str,
    upsample_factor: int = 1,
    transpose_result: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset from MATLAB files and apply preprocessing.

    Parameters
    ----------
    config : Config
        The configuration object. Used for the L3 filtering.
    data_dir : str
        The directory containing the data files.
    rsrp_file : str
        The filename of the RSRP data.
    sinr_file : str
        The filename of the SINR data.
    upsample_factor : int, optional
        The upsampling factor, by default 1.
    transpose_result : bool, optional
        Whether to transpose the result, by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The preprocessed RSRP and SINR data.

    """
    mat_rsrp = scipy.io.loadmat(os.path.join(data_dir, rsrp_file), squeeze_me=True)
    mat_sinr = scipy.io.loadmat(os.path.join(data_dir, sinr_file), squeeze_me=True)

    # Extract the data
    rsrp_key = list(mat_rsrp.keys())[-1]
    sinr_key = list(mat_sinr.keys())[-1]
    rsrp = mat_rsrp[rsrp_key].T
    sinr = mat_sinr[sinr_key].T

    # Upsampling
    n_steps, _ = rsrp.shape
    rsrp_up = scipy.signal.resample(rsrp, n_steps * upsample_factor, axis=0)
    sinr_up = scipy.signal.resample(sinr, n_steps * upsample_factor, axis=0)

    # L3 filtering
    rsrp_filtered = l3_filtering(rsrp_up, config.l3_filter_w)
    sinr_filtered = l3_filtering(sinr_up, config.l3_filter_w)

    if transpose_result:
        rsrp_filtered = rsrp_filtered.T
        sinr_filtered = sinr_filtered.T

    return rsrp_filtered, sinr_filtered


def l3_filtering(measurements: np.ndarray, w: float = 0.1) -> np.ndarray:
    """
    Implements L3 filtering with exponential smoothing.

    The formula used for the filtering is:

        R_filtered(t) = (1 - w) * R_filtered(t-1) + w * R_measured(t)

    where:
    - R_filtered(t) is the filtered value at time t.
    - w is the smoothing factor (0 < w < 1). A larger w gives more weight to recent measurements.
    - R_measured(t) is the measured value at time t.

    Parameters
    ----------
    measurements : np.ndarray
        A list or NumPy array of measured RSRP/SINR values.
    w : float, optional
        Smoothing factor (0 < w < 1), higher values give more weight to the current measurements.

    Returns
    -------
    np.ndarray
        The filtered RSRP/SINR values after applying L3 filtering.
    """
    filtered_values = np.zeros_like(measurements)  # Init

    filtered_values[0] = measurements[0]  # First filtered value = first measurement

    for t in range(1, len(measurements)):  # Apply the L3 filtering
        filtered_values[t] = (1 - w) * filtered_values[t - 1] + w * measurements[t]

    return filtered_values
