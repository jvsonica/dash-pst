from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show
from dslabs import plot_line_chart, plot_ts_multivariate_chart, ts_aggregation_by, HEIGHT


def run(df: DataFrame, target: str, gran_level: str='W', agg_funcs: str|dict ='mean', savefig=True):
    """Apply smoothing to df's target. Output a plot with multiple smoothing windows but
    return the series with `window` smoothing applied.

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing granularity.
        gran_level (str): Aggregation level to apply.
        savefig (bool, optional): Save generated figures to files. Defaults to True.

    Returns
        DataFrame: Aggregated DataFrame following passed parameters.
    """
    axs_original = plot_ts_multivariate_chart(df, title=f"target= {target}")
    fig_original = axs_original[0].get_figure()
    fig_original.tight_layout()

    # Perform aggregation
    agg_df: DataFrame = ts_aggregation_by(df, gran_level=gran_level, agg_func=agg_funcs)

    xas_aggregated = plot_ts_multivariate_chart(agg_df, title=f"after hourly aggregation {target}")
    fig_aggregated = xas_aggregated[0].get_figure()
    fig_aggregated.tight_layout()

    if savefig:
        fig_original.savefig(f'temp/{target}_aggregation_original.png')
        fig_aggregated.savefig(f'temp/{target}_aggregation_aggregated.png')
    else:
        fig_aggregated.show()

    return agg_df
