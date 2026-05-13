"""Normalisation and cell-cycle phase assignment for OMERO-Screen data.

This module is data-only: it parameterises the nucleus channel, normalises
DNA and EdU intensities, and assigns each cell to a phase. Plotting lives
in ``omero_screen_plots`` (in particular ``combplot_cellcycle``).

Typical usage example:
    import pandas as pd
    from omero_screen.cellcycle_analysis import cellcycle_analysis
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def cellcycle_analysis(
    df: pd.DataFrame,
    H3: bool = False,
    cyto: bool = True,
    nucleus_channel: str = "DAPI",
) -> pd.DataFrame:
    """Function to normalise cell cycle data and assign cell cycle phase for each cell line.

    Args:
        df: single cell data from omeroscreen
        H3: True if H3 data is present
        cyto: True if cytoplasmic data is present
        nucleus_channel: Name of the segmented nucleus channel as it appears in
            the feature column names (e.g. ``"DAPI"``, ``"Hoechst"``, ``"H2B_RFP"``).
            Used to locate the integrated intensity column
            ``integrated_int_{nucleus_channel}`` for DNA-content normalisation.
            The normalised column is written as
            ``integrated_int_{nucleus_channel}_norm``. Defaults to ``"DAPI"``
            for backward compatibility with legacy plates.

    Returns:
        dataframe with cell cycle and cell cycle detailed columns

    Raises:
        KeyError: If required columns are missing from the input DataFrame.
    """
    dna_col = f"integrated_int_{nucleus_channel}"
    dna_norm_col = f"{dna_col}_norm"

    # Validate required columns before processing
    required_cols = [
        dna_col,
        "intensity_mean_EdU_nucleus",
        "intensity_min_EdU_nucleus",
        "cell_line",
    ]
    if H3:
        required_cols.extend(
            [
                "intensity_mean_H3P_nucleus",
                "intensity_min_H3P_nucleus",
            ]
        )
    if cyto:
        required_cols.append("Cyto_ID")

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"Cell cycle analysis requires columns that are missing from the data: "
            f"{missing}. This usually means the corresponding channel "
            f"(EdU, DAPI, or H3P) was not included in the plate metadata, "
            f"or the channel name doesn't match expected naming."
        )

    df1 = df.copy()
    if H3:
        values = [
            dna_col,
            "intensity_mean_EdU_nucleus",
            "intensity_mean_H3P_nucleus",
        ]
        df1["intensity_mean_H3P_nucleus"] = (
            df1["intensity_mean_H3P_nucleus"]
            - df1["intensity_min_H3P_nucleus"]
            + 1
        )
    else:
        values = [dna_col, "intensity_mean_EdU_nucleus"]
    df1["intensity_mean_EdU_nucleus"] = (
        df1["intensity_mean_EdU_nucleus"]
        - df1["intensity_min_EdU_nucleus"]
        + 1
    )
    if cyto:
        df_agg = _agg_multinucleates(df1, dna_col=dna_col)
        df_agg_corr = _delete_duplicates(df_agg)
    else:
        df_agg_corr = df1.copy()
    tempfile = pd.DataFrame()
    for cell_line in df_agg_corr["cell_line"].unique():
        df1 = df_agg_corr.loc[df_agg_corr["cell_line"] == cell_line]
        df_norm = _normalise(df1, values)
        df_norm[dna_norm_col] = df_norm[dna_norm_col] * 2
        tempfile = pd.concat([tempfile, df_norm])
    return _assign_ccphase(data=tempfile, H3=H3, dna_norm_col=dna_norm_col)


# Helper Functions for cell cycle normalisation
def _agg_multinucleates(
    df: pd.DataFrame, dna_col: str = "integrated_int_DAPI"
) -> pd.DataFrame:
    """Function to aggregate multinucleates by summing up the nucleus area and DNA intensity.

    Args:
        df: single cell data from omeroscreen
        dna_col: Name of the integrated nucleus DNA-intensity column to sum-aggregate.

    Returns:
        corrected df with aggregated multinucleates
    """
    num_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
    str_cols = list(df.select_dtypes(include=["object"]).columns)
    # define the aggregation functions for each column
    agg_functions: dict[str, str | Callable[..., Any]] = {}
    sum_cols = {dna_col, "area_nucleus"}
    for col in num_cols:
        if col in sum_cols:
            agg_functions[col] = "sum"
        elif "max" in col and "nucleus" in col:
            agg_functions[col] = "max"
        elif "min" in col and "nucleus" in col:
            agg_functions[col] = "min"
        elif col == "label":
            agg_functions[col] = lambda x: tuple(x)
        else:
            agg_functions[col] = "mean"
    return df.groupby(str_cols + ["image_id", "Cyto_ID"], as_index=False).agg(
        agg_functions
    )


def _delete_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Function to delete duplicates from the agg_multinucleate dataframe.

    Args:
        df: dataframe from agg_multinucleates function
    Returns:
        df with deleted duplicates
    """
    temp_data = pd.DataFrame()
    for image in df["image_id"].unique():
        image_data = df.loc[df.image_id == image].drop_duplicates()
        temp_data = pd.concat([temp_data, image_data])
    return temp_data


def _normalise(df: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    """Data normalisation function.

    Identifies the most frequent intensity value and sets it to
    1 by division. For DAPI data this is set to two, to reflect diploid (2N) state of chromosomes.

    Args:
        df: dataframe from delete_duplicates function
        values: columns to normalise
    Returns:
        normalised data
    """
    norm_df = pd.DataFrame()
    for cell_line in df["cell_line"].unique():
        tmp_data = df.copy().loc[(df["cell_line"] == cell_line)]
        tmp_bins = 10000
        for value in values:
            y, x = np.histogram(tmp_data[value], bins=tmp_bins)
            max_value = x[np.where(y == np.max(y))]
            tmp_data[f"{value}_norm"] = tmp_data[value] / max_value[0]
        norm_df = pd.concat([norm_df, tmp_data])
    return norm_df


def _assign_ccphase(
    data: pd.DataFrame,
    H3: bool,
    dna_norm_col: str = "integrated_int_DAPI_norm",
) -> pd.DataFrame:
    """Assigns a cell cycle phase to each cell based on normalised EdU and DNA intensities.

    Args:
        data: dataframe from normalise function
        H3: True if H3 data is present
        dna_norm_col: Name of the normalised DNA-content column to threshold on.

    Returns:
        dataframe with cell cycle assignment
        (col: cellcycle (Sub-G1, G1, S, G2/M Polyploid)
        and col: cellcycle_detailed with Early S/Late S and Polyploid (non-replicating)
        Polyploid (replicating))
    """
    if H3:
        data["cell_cycle_detailed"] = data.apply(
            _thresholdingH3, axis=1, DAPI_col=dna_norm_col
        )
    else:
        data["cell_cycle_detailed"] = data.apply(
            _thresholding, axis=1, DAPI_col=dna_norm_col
        )
    data["cell_cycle"] = data["cell_cycle_detailed"]
    data["cell_cycle"] = data["cell_cycle"].replace(["Early S", "Late S"], "S")
    data["cell_cycle"] = data["cell_cycle"].replace(
        ["Polyploid (non-replicating)", "Polyploid (replicating)"], "Polyploid"
    )
    return data


def _thresholdingH3(
    data: pd.DataFrame,
    DAPI_col: str = "integrated_int_DAPI_norm",
    EdU_col: str = "intensity_mean_EdU_nucleus_norm",
    H3P_col: str = "intensity_mean_H3P_nucleus_norm",
) -> str:
    """Function to assign cell cycle phase based on thresholds of normalised EdU, DAPI and H3P intensities.

    Args:
        data: data from _assign_ccphase function
        DAPI_col: default 'integrated_int_DAPI_norm'
        EdU_col: default 'intensity_mean_EdU_nucleus_norm'
        H3P_col: default 'intensity_mean_H3P_nucleus_norm'
    Returns:
        string indicating cell cycle phase
    """
    if data[DAPI_col] <= 1.5:
        return "Sub-G1"

    elif 1.5 < data[DAPI_col] < 3 and data[EdU_col] < 3:
        return "G1"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] < 3 and data[H3P_col] < 5:
        return "G2"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] < 3 and data[H3P_col] > 5:
        return "M"

    elif 1.5 < data[DAPI_col] < 3 and data[EdU_col] > 3:
        return "Early S"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] > 3:
        return "Late S"

    elif data[DAPI_col] >= 5.5 and data[EdU_col] < 3:
        return "Polyploid (non-replicating)"

    elif data[DAPI_col] >= 5.5 and data[EdU_col] > 3:
        return "Polyploid (replicating)"

    else:
        return "Unassigned"


def _thresholding(
    data: pd.DataFrame,
    DAPI_col: str = "integrated_int_DAPI_norm",
    EdU_col: str = "intensity_mean_EdU_nucleus_norm",
) -> str:
    """Function to assign cell cycle phase based on thesholds of normalised EdU and DAPI intensities.

    Args:
        data: data from _assign_ccphase function
        DAPI_col: default 'integrated_int_DAPI_norm'
        EdU_col: default 'intensity_mean_EdU_nucleus_norm'
    Returns:
        string indicating cell cycle phase
    """
    if data[DAPI_col] <= 1.5:
        return "Sub-G1"

    elif 1.5 < data[DAPI_col] < 3 and data[EdU_col] < 3:
        return "G1"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] < 3:
        return "G2/M"

    elif 1.5 < data[DAPI_col] < 3 and data[EdU_col] > 3:
        return "Early S"

    elif 3 <= data[DAPI_col] < 5.5 and data[EdU_col] > 3:
        return "Late S"

    elif data[DAPI_col] >= 5.5 and data[EdU_col] < 3:
        return "Polyploid (non-replicating)"

    elif data[DAPI_col] >= 5.5 and data[EdU_col] > 3:
        return "Polyploid (replicating)"

    else:
        return "Unassigned"
