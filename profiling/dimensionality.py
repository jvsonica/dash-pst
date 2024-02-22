import pandas as pd
from matplotlib.pyplot import figure, show
from dslabs import plot_line_chart, HEIGHT

def analyze(df: pd.DataFrame, target: str, savefig = True):
    """Print out dimensionality analysis of `target` in the context of the timeseries
    dataset in `df`.
    Please ensure that your dataframe index is a DatetimeIndex. This means that 
    `isinstance(df.index, pd.DatetimeIndex)` should be truthy.

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing dimensionality
    """
    series: pd.Series = df[target]
    print(f"{target}:")
    print("Nr. Records = ", series.shape[0])
    print("First timestamp", series.index[0])
    print("Last timestamp", series.index[-1])

    f = figure(figsize=(3 * HEIGHT, HEIGHT / 2))
    plot_line_chart(
        series.index.to_list(),
        series.to_list(),
        xlabel=series.index.name,
        ylabel=target,
        title=f"{target}"
    )
    f.tight_layout()

    if (savefig):
        f.savefig(f'temp/{target}_dimensionality.png')
    else:
        show()
