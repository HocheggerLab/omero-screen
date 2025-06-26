from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def cell_cycle_data():
    """Load real cell cycle data for testing

    Dataset properties:
    - Contains conditions: ['DMSO', 'Drug1', ...]
    - Features: DAPI, EdU intensities
    - Cell cycle phases
    """
    data_path = Path(__file__).parent / "../../../examples/plots_test_data.csv"
    return pd.read_csv(data_path)


@pytest.fixture
def filtered_data(cell_cycle_data):
    """Pre-filtered dataset with specific conditions"""
    conditions = ["ctr", "palb"]
    data = cell_cycle_data[cell_cycle_data.condition.isin(conditions)].copy()

    # Add normalized columns expected by some plots if they don't exist
    if 'integrated_int_DAPI_norm' not in data.columns:
        # Use an existing numeric column as proxy
        if 'intensity_mean_DAPI_nucleus' in data.columns:
            data['integrated_int_DAPI_norm'] = data['intensity_mean_DAPI_nucleus'] / data['intensity_mean_DAPI_nucleus'].mean()
        else:
            data['integrated_int_DAPI_norm'] = 1.0

    if 'intensity_mean_EdU_nucleus_norm' not in data.columns:
        if 'intensity_mean_EdU_nucleus' in data.columns:
            data['intensity_mean_EdU_nucleus_norm'] = data['intensity_mean_EdU_nucleus'] / data['intensity_mean_EdU_nucleus'].mean()
        else:
            data['intensity_mean_EdU_nucleus_norm'] = 1.0

    return data


@pytest.fixture
def signficance_data(cell_cycle_data):
    """engneer three replicates of data with significant and non-significant results"""
    # TODO Make two copies of the df and Take data from the three condition categories
    # TODO Add a significant result to one and a non-significant result to the other
    # TODO Return a df with the two dfs concatenated

    return cell_cycle_data
