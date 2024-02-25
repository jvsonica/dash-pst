from pandas import DataFrame
from dslabs import plot_ts_multivariate_chart, ts_aggregation_by
from preprocess import aggregation_func_by_col


def analyze(df: DataFrame, target: str, gran_level: str='W', agg_funcs: str|dict = aggregation_func_by_col, savefig=True):
    """Analyze smoothing to df's target. Output a plot with multiple smoothing windows but
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
    agg_funcs = { k: v for (k,v) in aggregation_func_by_col.items() if k in df.columns }
    agg_df: DataFrame = ts_aggregation_by(df, gran_level=gran_level, agg_func=agg_funcs)

    xas_aggregated = plot_ts_multivariate_chart(agg_df, title=f"after hourly aggregation {target}")
    fig_aggregated = xas_aggregated[0].get_figure()
    fig_aggregated.tight_layout()

    if savefig:
        fig_original.savefig(f'temp/{target}_aggregation_original.png')
        fig_aggregated.savefig(f'temp/{target}_aggregation_aggregated.png')


def run(df: DataFrame, gran_level: str='W', agg_funcs=aggregation_func_by_col):
    """Apply smoothing to the input `df` following `gran_level and `agg_funcs`.

    Args:
        df (DataFrame): 
        target (str): 
        gran_level (str, optional): Defaults to 'W'.
        agg_funcs (str | dict, optional): Defaults to 'mean'.
    """
    agg_funcs = { k: v for (k,v) in aggregation_func_by_col.items() if k in df.columns }
    return ts_aggregation_by(df, gran_level=gran_level, agg_func=agg_funcs)
