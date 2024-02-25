from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from dslabs import plot_line_chart, HEIGHT
from matplotlib.pyplot import figure, show


def scale_all_dataframe(data: DataFrame) -> DataFrame:
    vars: list[str] = data.columns.to_list()
    transf: StandardScaler = StandardScaler().fit(data)
    df = DataFrame(transf.transform(data), index=data.index)
    df.columns = vars
    return df


def analyze(df: DataFrame, target: str, savefig=True):
    """Apply scaling to all columns of the dataframe, including target variable.

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing granularity
        savefig (bool, optional): Save generated figures to files. Defaults to True.

    Returns
        DataFrame: Scaled dataframe.
    """
    scaled: DataFrame = scale_all_dataframe(df)

    ss: Series = scaled[target]
    fig = figure(figsize=(3 * HEIGHT, HEIGHT / 2))
    plot_line_chart(
        ss.index.to_list(),
        ss.to_list(),
        xlabel=ss.index.name,
        ylabel=target,
        title=f"{target} after scaling",
    )
    fig.tight_layout()
    if savefig:
        fig.savefig(f'temp/{target}_scaling.png')

    return scaled


def run(df: DataFrame):
    """Apply StandardScaler to entire dataframe.
    TODO: explain scaling

    Args:
        df (DataFrame):

    Returns:
        DataFrame: 
    """
    return scale_all_dataframe(df)
