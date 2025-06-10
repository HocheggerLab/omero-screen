import matplotlib.pyplot as plt

from omero_screen_plots.cellcycleplot import cc_phase, cellcycle_plot


def test_cc_phase(filtered_data):
    df_cc = cc_phase(filtered_data)
    assert df_cc.shape == (53, 5)
    assert df_cc.percent.mean() == 22.641509433962263


def test_cellcycle_plots(filtered_data):
    conditions = ["ctr", "palb"]
    cellcycle_plot(
        df=filtered_data,
        conditions=conditions,
        selector_col="cell_line",
        selector_val="RPE1wt",
        condition_col="condition",
        save=False,
    )
    plt.close("all")
