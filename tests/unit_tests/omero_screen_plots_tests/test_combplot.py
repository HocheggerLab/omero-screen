import matplotlib.pyplot as plt

from omero_screen_plots.combplot import comb_plot


def test_combplot_with_real_data(filtered_data, tmp_path):
    """Test combplot using real cell cycle data"""
    conditions = ["ctr", "palb"]
    feature_col = "intensity_mean_yH2AX_nucleus"

    comb_plot(
        df=filtered_data,
        conditions=conditions,
        feature_col=feature_col,
        feature_y_lim=8000,
        condition_col="condition",
        selector_col="cell_line",
        selector_val="RPE1wt",
        title="test",
        save=False,
        path=None,
    )
    plt.close("all")
