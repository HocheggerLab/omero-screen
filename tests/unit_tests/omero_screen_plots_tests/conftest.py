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
    return cell_cycle_data[cell_cycle_data.condition.isin(conditions)]


@pytest.fixture
def signficance_data(cell_cycle_data):
    """engneer three replicates of data with significant and non-significant results"""
    # TODO Make two copies of the df and Take data from the three condition categories
    # TODO Add a significant result to one and a non-significant result to the other
    # TODO Return a df with the two dfs concatenated

    return cell_cycle_data
