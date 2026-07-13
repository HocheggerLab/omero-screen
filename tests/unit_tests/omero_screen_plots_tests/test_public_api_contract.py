"""Public-API contract tests for omero_screen_plots.

This is the Phase 0 safety net for the plot-architecture refactor. The team's
only stable contract is the set of ``*_api`` functions exported from the package
top level. These tests pin two things so the internal refactor cannot silently
break callers:

1. **Signatures** — the exact ``inspect.signature`` of every public function
   (and the ``__all__`` export set) is compared against a committed golden file
   (``public_api_signatures.json``). Any added/removed/renamed parameter or
   changed default fails the test.
2. **Return shapes** — each public function is called on synthetic data and the
   returned object's shape is asserted (bare ``Axes`` vs ``list[Axes]`` vs bare
   ``Figure``), including the condition-count-dependent shapes of
   ``histogram_plot`` / ``scatter_plot`` that callers unpack.

If you intentionally change the public API, regenerate the golden with::

    uv run python -c "import inspect, json, omero_screen_plots as osp; \
        funcs=['count_plot','feature_plot','feature_norm_plot','cellcycle_plot',\
        'cellcycle_stacked','classification_plot','histogram_plot','scatter_plot',\
        'combplot_feature','combplot_cellcycle','well_qc_plot']; \
        g={n:str(inspect.signature(getattr(osp,n))) for n in funcs}; \
        g['__all__']=sorted(osp.__all__); \
        open('tests/unit_tests/omero_screen_plots_tests/public_api_signatures.json','w')\
        .write(json.dumps(g, indent=2))"

and review the diff — a change here means downstream user code may need updating.
"""

import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no display needed

import matplotlib.pyplot as plt
import numpy as np
import omero_screen_plots as osp
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

GOLDEN_PATH = Path(__file__).parent / "public_api_signatures.json"

# The public callable surface. Kept explicit (not derived from __all__) so that
# adding a public plot function is a deliberate edit to this list + the golden.
PUBLIC_FUNCTIONS = [
    "count_plot",
    "feature_plot",
    "feature_norm_plot",
    "cellcycle_plot",
    "cellcycle_stacked",
    "classification_plot",
    "histogram_plot",
    "scatter_plot",
    "combplot_feature",
    "combplot_cellcycle",
    "well_qc_plot",
]


@pytest.fixture(scope="module")
def golden() -> dict[str, object]:
    """Load the committed golden signature snapshot."""
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test to keep memory flat."""
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# 1. Signature / export contract
# --------------------------------------------------------------------------- #


def test_all_exports_unchanged(golden: dict[str, object]) -> None:
    """The __all__ export set must not drift silently."""
    assert sorted(osp.__all__) == golden["__all__"]


def test_public_functions_are_exported() -> None:
    """Every function we contract-test is actually importable from the top level."""
    for name in PUBLIC_FUNCTIONS:
        assert hasattr(osp, name), (
            f"{name} is not exported from omero_screen_plots"
        )
        assert callable(getattr(osp, name))


@pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
def test_signature_unchanged(name: str, golden: dict[str, object]) -> None:
    """Each public function's signature must match the committed golden."""
    current = str(inspect.signature(getattr(osp, name)))
    assert current == golden[name], (
        f"Signature of {name} changed.\n"
        f"  golden : {golden[name]}\n"
        f"  current: {current}\n"
        "If this change is intentional, regenerate public_api_signatures.json "
        "(see module docstring) and review the diff for downstream impact."
    )


# --------------------------------------------------------------------------- #
# 2. Return-shape contract
# --------------------------------------------------------------------------- #


@pytest.fixture
def full_plate_data() -> pd.DataFrame:
    """Synthetic dataset exercising every public plot function.

    3 plates x 3 conditions (>=3 plates so significance paths run), with cell
    cycle, DNA/EdU normalised columns, generic features, and a Class column.
    """
    rng = np.random.default_rng(42)
    phases = ["Sub-G1", "G1", "S", "G2/M", "Polyploid"]
    classes = ["negative", "positive"]
    rows = []
    mid = 1
    for plate_id in (1001, 1002, 1003):
        for condition in ("control", "treatment1", "treatment2"):
            for well in ("A1", "A2"):
                for _ in range(30):
                    dna = float(rng.uniform(1.8, 4.2))
                    rows.append(
                        {
                            "plate_id": plate_id,
                            "well": well,
                            "well_id": mid * 10,
                            "image_id": mid * 100,
                            "experiment": f"exp_{plate_id}_{well}_{mid}",
                            "condition": condition,
                            "cell_line": "MCF10A",
                            "cell_cycle": phases[mid % len(phases)],
                            "Class": classes[mid % len(classes)],
                            "integrated_int_DAPI_norm": dna,
                            "intensity_mean_EdU_nucleus_norm": float(
                                rng.uniform(1, 50)
                            ),
                            "intensity_mean_EdU_nucleus": float(
                                rng.uniform(1000, 50000)
                            ),
                            "area_nucleus": float(rng.uniform(100, 500)),
                            "intensity_mean_p21_nucleus": float(
                                rng.uniform(100, 8000)
                            ),
                            "intensity_mean_DAPI_nucleus": dna * 10000,
                        }
                    )
                    mid += 1
    return pd.DataFrame(rows)


CONDITIONS = ["control", "treatment1", "treatment2"]


def _assert_fig_axes(result: object) -> None:
    """Assert (Figure, Axes)."""
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def _assert_fig_axes_list(result: object) -> None:
    """Assert (Figure, list[Axes])."""
    assert isinstance(result, tuple) and len(result) == 2
    fig, axes = result
    assert isinstance(fig, Figure)
    assert isinstance(axes, list) and len(axes) >= 1
    assert all(isinstance(a, Axes) for a in axes)


def test_count_plot_shape(full_plate_data: pd.DataFrame) -> None:
    """count_plot returns (Figure, Axes)."""
    _assert_fig_axes(
        osp.count_plot(
            df=full_plate_data,
            norm_control="control",
            conditions=CONDITIONS,
            selector_col=None,
            save=False,
        )
    )


def test_feature_plot_shape(full_plate_data: pd.DataFrame) -> None:
    """feature_plot returns (Figure, Axes)."""
    _assert_fig_axes(
        osp.feature_plot(
            df=full_plate_data,
            feature="area_nucleus",
            conditions=CONDITIONS,
            selector_col=None,
            save=False,
        )
    )


def test_feature_norm_plot_shape(full_plate_data: pd.DataFrame) -> None:
    """feature_norm_plot returns (Figure, Axes)."""
    _assert_fig_axes(
        osp.feature_norm_plot(
            df=full_plate_data,
            feature="intensity_mean_p21_nucleus",
            conditions=CONDITIONS,
            selector_col=None,
            save=False,
        )
    )


def test_cellcycle_plot_shape(full_plate_data: pd.DataFrame) -> None:
    """cellcycle_plot returns (Figure, list[Axes]) — one subplot per phase."""
    # cellcycle_plot returns a LIST of axes (one subplot per phase).
    _assert_fig_axes_list(
        osp.cellcycle_plot(
            df=full_plate_data,
            conditions=CONDITIONS,
            selector_col=None,
            save=False,
        )
    )


def test_cellcycle_stacked_shape(full_plate_data: pd.DataFrame) -> None:
    """cellcycle_stacked returns (Figure, Axes)."""
    _assert_fig_axes(
        osp.cellcycle_stacked(
            df=full_plate_data,
            conditions=CONDITIONS,
            selector_col=None,
            save=False,
        )
    )


def test_classification_plot_shape(full_plate_data: pd.DataFrame) -> None:
    """classification_plot returns (Figure, Axes)."""
    _assert_fig_axes(
        osp.classification_plot(
            df=full_plate_data,
            classes=["negative", "positive"],
            conditions=CONDITIONS,
            selector_col=None,
            save=False,
        )
    )


def test_histogram_plot_single_condition_returns_axes(
    full_plate_data: pd.DataFrame,
) -> None:
    """histogram_plot with a single condition returns (Figure, Axes)."""
    # Single condition -> bare Axes (callers do `fig, ax = ...`).
    _assert_fig_axes(
        osp.histogram_plot(
            df=full_plate_data,
            feature="area_nucleus",
            conditions="control",
            save=False,
        )
    )


def test_histogram_plot_multi_condition_returns_list(
    full_plate_data: pd.DataFrame,
) -> None:
    """histogram_plot with multiple conditions returns (Figure, list[Axes])."""
    # Multiple conditions -> list[Axes].
    _assert_fig_axes_list(
        osp.histogram_plot(
            df=full_plate_data,
            feature="area_nucleus",
            conditions=CONDITIONS,
            save=False,
        )
    )


def test_scatter_plot_single_condition_returns_axes(
    full_plate_data: pd.DataFrame,
) -> None:
    """scatter_plot with a single condition returns (Figure, Axes)."""
    _assert_fig_axes(
        osp.scatter_plot(
            df=full_plate_data,
            conditions="control",
            selector_col=None,
            save=False,
        )
    )


def test_scatter_plot_multi_condition_returns_list(
    full_plate_data: pd.DataFrame,
) -> None:
    """scatter_plot with multiple conditions returns (Figure, list[Axes])."""
    _assert_fig_axes_list(
        osp.scatter_plot(
            df=full_plate_data,
            conditions=CONDITIONS,
            selector_col=None,
            save=False,
        )
    )


def test_combplot_feature_shape(full_plate_data: pd.DataFrame) -> None:
    """combplot_feature returns (Figure, list[Axes])."""
    _assert_fig_axes_list(
        osp.combplot_feature(
            df=full_plate_data,
            conditions=CONDITIONS,
            feature="intensity_mean_p21_nucleus",
            selector_col=None,
            save=False,
        )
    )


def test_combplot_cellcycle_shape(full_plate_data: pd.DataFrame) -> None:
    """combplot_cellcycle returns (Figure, list[Axes])."""
    _assert_fig_axes_list(
        osp.combplot_cellcycle(
            df=full_plate_data,
            conditions=CONDITIONS,
            selector_col=None,
            save=False,
        )
    )


def test_well_qc_plot_returns_bare_figure(
    full_plate_data: pd.DataFrame,
) -> None:
    """well_qc_plot returns a bare Figure (not a tuple)."""
    # well_qc_plot is the odd one out: it returns a bare Figure, not a tuple.
    result = osp.well_qc_plot(df=full_plate_data, save=False)
    assert isinstance(result, Figure)
