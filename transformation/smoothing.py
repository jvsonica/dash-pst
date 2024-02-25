from pandas import Series
from dslabs import plot_line_chart, HEIGHT
from matplotlib.pyplot import Figure, Axes, subplots
from preprocess import aggregation_func_by_col


def analyze(series: Series, target: str, window=50, savefig=True):
    """Apply smoothing a series. Output a plot with multiple smoothing windows but
    return the series with `window` smoothing applied.

    Args:
        df (Series): Series to apply smoothing. 
        target (str): Column name of `df` from which we are analyzing granularity.
        window (int): Smoothing window to apply.
        savefig (bool, optional): Save generated figures to files. Defaults to True.

    Returns
        Series: Scaled target variable.
    """
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
        fig.savefig(f'temp/{target}_smoothing_mean.png')

    agg_func = aggregation_func_by_col[series.name]
    return series.rolling(window=window).agg(agg_func)


def run(series: Series, window:int = 50, agg_func:str = 'mean'):
    """Apply `agg_func` smoothing on `window`.

    Args:
        series (Series): _description_
        target (str): _description_
        window (int, optional): _description_. Defaults to 50.
        agg_func (str, optional): _description_. Defaults to 'mean'.

    Returns:
        Series: _description_
    """
    return series.rolling(window=window).agg(agg_func)
