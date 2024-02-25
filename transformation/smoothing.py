from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from dslabs import plot_line_chart, HEIGHT
from matplotlib.pyplot import Figure, Axes, subplots, figure


def run(df: DataFrame, target: str, window=50, savefig=True):
    """Apply smoothing to df's target. Output a plot with multiple smoothing windows but
    return the series with `window` smoothing applied.

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing granularity.
        window (int): Smoothing window to apply.
        savefig (bool, optional): Save generated figures to files. Defaults to True.

    Returns
        Series: Scaled target variable.
    """
    series: Series = df[target]
    sizes: list[int] = [25, 50, 75, 100]
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
    fig.suptitle(f"{target} after smoothing")

    for i in range(len(sizes)):
        ss_smooth: Series = series.rolling(window=sizes[i]).mean()
        plot_line_chart(
            ss_smooth.index.to_list(),
            ss_smooth.to_list(),
            ax=axs[i],
            xlabel=ss_smooth.index.name,
            ylabel=target,
            title=f"size={sizes[i]}",
        )
    fig.tight_layout()
    if savefig:
        fig.savefig(f'temp/{target}_smoothing.png')
    else:
        fig.show()

    return series.rolling(window=window).mean()