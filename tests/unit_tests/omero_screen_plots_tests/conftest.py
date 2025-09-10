"""Shared fixtures for omero-screen-plots tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_plate_data():
    """Create a synthetic dataset with realistic plate data structure.

    Creates a comprehensive dataset with:
    - 3 plates (1001, 1002, 1003)
    - 3 conditions (control, treatment1, treatment2)
    - 2-3 wells per plate/condition combination
    - 5-10 rows per well (simulating multiple cells/experiments)
    - 2 cell lines (MCF10A, HeLa)
    - Various measurement columns for testing different features

    Returns:
        pd.DataFrame: Synthetic plate data with all required columns
    """
    np.random.seed(42)  # For reproducible results

    data = []
    plates = [1001, 1002, 1003]
    conditions = ["control", "treatment1", "treatment2"]
    wells = ["A1", "A2", "B1", "B2", "C1"]
    cell_lines = ["MCF10A", "HeLa"]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            # Use 2-3 wells per condition
            wells_for_condition = np.random.choice(wells, size=np.random.randint(2, 4), replace=False)

            for well in wells_for_condition:
                # 5-10 experiments per well
                n_experiments = np.random.randint(5, 11)

                for _ in range(n_experiments):
                    cell_line = np.random.choice(cell_lines)

                    row = {
                        "plate_id": plate_id,
                        "well": well,
                        "experiment": f"exp_{plate_id}_{well}_{measurement_id}",
                        "condition": condition,
                        "cell_line": cell_line,
                        "measurement_id": measurement_id,
                        "well_id": measurement_id * 10,
                        "image_id": measurement_id * 100,
                        # Add various measurement columns for feature testing
                        "area_nucleus": np.random.uniform(100, 500),
                        "area_cell": np.random.uniform(200, 800),
                        "intensity_mean_DAPI_nucleus": np.random.uniform(1000, 20000),
                        "intensity_mean_GFP_cell": np.random.uniform(500, 15000),
                        "intensity_median_DAPI_nucleus": np.random.uniform(800, 18000),
                        "perimeter_nucleus": np.random.uniform(50, 150),
                        "eccentricity_nucleus": np.random.uniform(0.1, 0.9),
                        "solidity_nucleus": np.random.uniform(0.7, 1.0),
                        # Add some correlated features for realistic testing
                        "aspect_ratio": np.random.uniform(1.0, 3.0),
                        "roundness": np.random.uniform(0.3, 1.0),
                    }

                    # Make some features condition-dependent for testing significance
                    if condition == "treatment1":
                        row["area_nucleus"] *= 1.2  # Slightly larger nuclei
                        row["intensity_mean_DAPI_nucleus"] *= 0.8  # Lower intensity
                    elif condition == "treatment2":
                        row["area_nucleus"] *= 0.9  # Slightly smaller nuclei
                        row["intensity_mean_DAPI_nucleus"] *= 1.3  # Higher intensity

                    data.append(row)
                    measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def minimal_plate_data():
    """Create minimal dataset for basic testing.

    Returns:
        pd.DataFrame: Minimal dataset with just required columns
    """
    data = [
        {
            "plate_id": 1001,
            "well": "A1",
            "experiment": "exp1",
            "condition": "control",
            "cell_line": "MCF10A",
            "area_nucleus": 250.0,
            "intensity_mean_DAPI_nucleus": 10000.0,
        },
        {
            "plate_id": 1001,
            "well": "A1",
            "experiment": "exp2",
            "condition": "control",
            "cell_line": "MCF10A",
            "area_nucleus": 300.0,
            "intensity_mean_DAPI_nucleus": 12000.0,
        },
        {
            "plate_id": 1001,
            "well": "A2",
            "experiment": "exp3",
            "condition": "treatment1",
            "cell_line": "MCF10A",
            "area_nucleus": 280.0,
            "intensity_mean_DAPI_nucleus": 9000.0,
        },
        {
            "plate_id": 1002,
            "well": "A1",
            "experiment": "exp4",
            "condition": "control",
            "cell_line": "MCF10A",
            "area_nucleus": 320.0,
            "intensity_mean_DAPI_nucleus": 11000.0,
        },
        {
            "plate_id": 1002,
            "well": "A2",
            "experiment": "exp5",
            "condition": "treatment1",
            "cell_line": "MCF10A",
            "area_nucleus": 270.0,
            "intensity_mean_DAPI_nucleus": 8500.0,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def zero_control_data():
    """Create data where control condition results in zero counts."""
    data = [
        {
            "plate_id": 1001,
            "well": "A1",
            "experiment": "exp1",
            "condition": "treatment1",
            "area_nucleus": 250.0,
            "intensity_mean_DAPI_nucleus": 10000.0,
        },
        {
            "plate_id": 1001,
            "well": "A2",
            "experiment": "exp2",
            "condition": "treatment1",
            "area_nucleus": 300.0,
            "intensity_mean_DAPI_nucleus": 12000.0,
        },
        {
            "plate_id": 1002,
            "well": "A1",
            "experiment": "exp3",
            "condition": "treatment1",
            "area_nucleus": 280.0,
            "intensity_mean_DAPI_nucleus": 9000.0,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def single_condition_data():
    """Create data with only one condition."""
    data = [
        {
            "plate_id": 1001,
            "well": "A1",
            "experiment": "exp1",
            "condition": "control",
            "area_nucleus": 250.0,
            "intensity_mean_DAPI_nucleus": 10000.0,
        },
        {
            "plate_id": 1001,
            "well": "A2",
            "experiment": "exp2",
            "condition": "control",
            "area_nucleus": 300.0,
            "intensity_mean_DAPI_nucleus": 12000.0,
        },
        {
            "plate_id": 1002,
            "well": "A1",
            "experiment": "exp3",
            "condition": "control",
            "area_nucleus": 280.0,
            "intensity_mean_DAPI_nucleus": 9000.0,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def many_plates_data():
    """Create data with many plates for significance testing."""
    data = []
    for plate_id in range(1001, 1006):  # 5 plates
        for condition in ["control", "treatment1"]:
            for well in ["A1", "A2"]:
                for exp_num in range(3):  # 3 experiments per well
                    base_area = 250.0 if condition == "control" else 300.0
                    base_intensity = 10000.0 if condition == "control" else 8000.0

                    data.append({
                        "plate_id": plate_id,
                        "well": well,
                        "experiment": f"exp_{plate_id}_{well}_{exp_num}",
                        "condition": condition,
                        "area_nucleus": base_area + np.random.normal(0, 50),
                        "intensity_mean_DAPI_nucleus": base_intensity + np.random.normal(0, 1000),
                    })

    return pd.DataFrame(data)


@pytest.fixture
def scaled_feature_data():
    """Create data with features that need scaling (wide value ranges)."""
    np.random.seed(123)
    data = []

    for plate_id in [1001, 1002]:
        for condition in ["control", "treatment1"]:
            for i in range(10):
                data.append({
                    "plate_id": plate_id,
                    "well": f"A{i+1}",
                    "experiment": f"exp_{plate_id}_{condition}_{i}",
                    "condition": condition,
                    "cell_line": "MCF10A",
                    # Wide range feature (0-65535 range, simulating 16-bit intensity)
                    "intensity_raw": np.random.uniform(0, 65535),
                    # Normal range feature
                    "area_nucleus": np.random.uniform(100, 500),
                })

    return pd.DataFrame(data)


@pytest.fixture
def dna_edu_data():
    """Create synthetic DNA/EdU data for cell cycle analysis testing."""
    np.random.seed(42)
    data = []

    plates = [1001, 1002]
    conditions = ["control", "treatment1"]

    measurement_id = 1

    for plate_id in plates:
        for condition in conditions:
            # Generate realistic DNA content and EdU intensity data
            for _ in range(50):  # 50 cells per condition
                # DNA content (log-normal distribution around 2N and 4N)
                if np.random.random() < 0.3:  # 30% in S phase
                    dna_content = np.random.uniform(2.2, 3.8)  # S phase
                    edu_intensity = np.random.uniform(1000, 8000)  # EdU positive
                    cell_cycle = "S"
                elif np.random.random() < 0.6:  # 60% of remaining in G1
                    dna_content = np.random.normal(2.0, 0.1)  # G1 phase
                    edu_intensity = np.random.uniform(50, 500)  # EdU negative
                    cell_cycle = "G1"
                else:  # Remaining in G2/M
                    dna_content = np.random.normal(4.0, 0.2)  # G2/M phase
                    edu_intensity = np.random.uniform(50, 500)  # EdU negative
                    cell_cycle = "G2/M"

                # Add some noise and ensure positive values
                dna_content = max(1.0, dna_content + np.random.normal(0, 0.1))
                edu_intensity = max(10, edu_intensity + np.random.normal(0, 100))

                data.append({
                    "plate_id": plate_id,
                    "well": f"A{measurement_id % 12 + 1}",
                    "experiment": f"exp_{measurement_id}",
                    "condition": condition,
                    "cell_line": "MCF10A",
                    "measurement_id": measurement_id,
                    "integrated_int_DAPI_norm": dna_content,
                    "intensity_mean_EdU_nucleus_norm": edu_intensity,
                    "intensity_mean_EdU_nucleus": edu_intensity * 1000,  # Non-normalized version
                    "cell_cycle": cell_cycle,
                    # Additional features for testing
                    "area_nucleus": np.random.uniform(100, 500),
                    "area_cell": np.random.uniform(200, 800),
                    "intensity_mean_p21_nucleus": np.random.uniform(100, 5000),
                    "intensity_mean_DAPI_nucleus": dna_content * 10000,
                })
                measurement_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def threshold_test_data():
    """Create data for threshold testing with clear above/below threshold values."""
    data = []

    # Create data with clear threshold separation
    for i in range(50):
        data.append({
            "plate_id": 1001,
            "well": "A1",
            "experiment": f"exp_{i}",
            "condition": "control",
            "cell_line": "MCF10A",
            "integrated_int_DAPI_norm": 2.0 + np.random.normal(0, 0.1),
            "intensity_mean_p21_nucleus": 2000 + np.random.normal(0, 200),  # Below threshold
            "area_nucleus": np.random.uniform(100, 300),
        })

    for i in range(50, 100):
        data.append({
            "plate_id": 1001,
            "well": "A2",
            "experiment": f"exp_{i}",
            "condition": "treatment1",
            "cell_line": "MCF10A",
            "integrated_int_DAPI_norm": 2.0 + np.random.normal(0, 0.1),
            "intensity_mean_p21_nucleus": 8000 + np.random.normal(0, 500),  # Above threshold
            "area_nucleus": np.random.uniform(100, 300),
        })

    return pd.DataFrame(data)


@pytest.fixture
def histogram_test_data():
    """Create data specifically for histogram testing with various distributions."""
    np.random.seed(123)
    data = []

    conditions = ["control", "treatment1", "treatment2"]

    measurement_id = 1

    for condition in conditions:
        # Create different distribution shapes for each condition
        if condition == "control":
            # Normal distribution
            feature_values = np.random.normal(1000, 200, 100)
        elif condition == "treatment1":
            # Log-normal distribution
            feature_values = np.random.lognormal(np.log(1000), 0.5, 100)
        else:  # treatment2
            # Bimodal distribution
            mode1 = np.random.normal(800, 100, 50)
            mode2 = np.random.normal(1500, 150, 50)
            feature_values = np.concatenate([mode1, mode2])

        for value in feature_values:
            data.append({
                "plate_id": 1001,
                "well": f"A{measurement_id % 12 + 1}",
                "experiment": f"exp_{measurement_id}",
                "condition": condition,
                "cell_line": "MCF10A",
                "measurement_id": measurement_id,
                # Feature for histogram testing
                "test_feature": max(10, value),  # Ensure positive values
                # DNA content feature for log-scale testing
                "integrated_int_DAPI_norm": np.random.uniform(1.5, 4.5),
                # Wide range feature for binning tests
                "wide_range_feature": np.random.uniform(0, 65535),
            })
            measurement_id += 1

    return pd.DataFrame(data)
