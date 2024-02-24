from pandas import Series, DataFrame, Index, Period
from matplotlib.pyplot import figure, show, subplots, Axes, Figure
from dslabs import plot_line_chart, HEIGHT

def ts_aggregation_by(
    data: Series | DataFrame,
    gran_level: str = "D",
    agg_func: str = "mean",
) -> Series | DataFrame:
    df: Series | DataFrame = data.copy()
    index: Index[Period] = df.index.to_period(gran_level)
    df = df.groupby(by=index, dropna=True, sort=True).agg(agg_func)
    df.index.drop_duplicates()
    df.index = df.index.to_timestamp()

    return df


def analyze(df: DataFrame, target: str, savefig = True):
    """Print out granularity analysis of `target` in the context of the timeseries
    dataset in `df`.
    Please ensure that your dataframe index is a DatetimeIndex. This means that 
    `isinstance(df.index, pd.DatetimeIndex)` should be truthy.

    In this analysis we aggregate data by day, by week and by month to find smoother
    versions of our time series, with less noise. 

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing granularity
    """
    print('\n-- Granularity --')
    series = df[target]
    grans: list[str] = ["D", "W", "M"]

    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(len(grans), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(grans)))
    fig.suptitle(f"{target} aggregation study")

    for i in range(len(grans)):
        ss: Series = ts_aggregation_by(series, grans[i])
        plot_line_chart(
            ss.index.to_list(),
            ss.to_list(),
            ax=axs[i],
            xlabel=f"{ss.index.name} ({grans[i]})",
            ylabel=target,
            title=f"granularity={grans[i]}",
        )
    fig.tight_layout()

    if (savefig):
        fig.savefig(f'temp/{target}_granularity.png')
        print(f'saved temp/{target}_granularity.png')
    else:
        show()
