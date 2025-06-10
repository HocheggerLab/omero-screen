import matplotlib.pyplot as plt

from omero_screen_plots.countplot import count_plot, norm_count


def test_norm_count(filtered_data):
    df = norm_count(filtered_data, norm_control="ctr")
    assert df["normalized_count"].sum() == 6.117847659139087


def test_count_fig(filtered_data):
    count_plot(
        filtered_data,
        norm_control="ctr",
        conditions=["ctr", "palb"],
        selector_col="cell_line",
        selector_val="RPE1wt",
        title="test",
        save=False,
    )
    plt.close("all")
