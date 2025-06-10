import matplotlib.pyplot as plt

from omero_screen_plots.featureplot import feature_plot

conditions = ["ctr", "plab"]


def test_feature_plot(filtered_data):
    feature_plot(
        df=filtered_data,
        feature="area_cell",
        conditions=conditions,
        selector_col="cell_line",
        selector_val="RPE1p53KO",
        title="test",
        save=False,
    )
    plt.close("all")
