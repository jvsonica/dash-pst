from pandas import DataFrame, Series
from matplotlib.pyplot import subplots
from dslabs import ts_aggregation_by, plot_forecasting_series_on_ax
from preprocess import aggregation_func_by_col
from pipelines import linear_regression
from pipelines.tasks.evaluate import evaluate


def compare_with_linear_reg(df, target, ax=None, plot_subtitle=""):
    train, test, predicted_train, predicted_test = linear_regression.run(df, target, None, path=None)
    plot_forecasting_series_on_ax(
        train,
        test,
        predicted_test,
        ax=ax,
        title=plot_subtitle,
        xlabel=train.index.name,
        ylabel=""
    )
    return evaluate(train, test, predicted_train, predicted_test)

def analyze(df: DataFrame, target: str, agg_funcs: str|dict = aggregation_func_by_col, savefig=True):
    """Analyze smoothing to df's target. Output plots with multiple aggregation windows
    and compare their results by applying Linear Regression on each.

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing granularity.
        savefig (bool, optional): Save generated figures to files. Defaults to True.

    Returns
        DataFrame: Aggregated DataFrame following passed parameters.
    """
    # Perform aggregations
    agg_funcs = { k: v for (k,v) in aggregation_func_by_col.items() if k in df.columns }
    hour: DataFrame = ts_aggregation_by(df, gran_level='h', agg_func=agg_funcs).dropna()
    day: DataFrame = ts_aggregation_by(df, gran_level='d', agg_func=agg_funcs).dropna()

    # Setup result plot
    fig, axs = subplots(3, 1, figsize=(16, 3 * 1.5))
    fig.suptitle(f"Aggregation Analysis on {target}")

    # No Agg
    metrics_raw = compare_with_linear_reg(df, target, ax=axs[0], plot_subtitle="no aggregation")
    print('No aggregation:')
    print(DataFrame(metrics_raw))

    # rule: h
    hour: DataFrame = ts_aggregation_by(df, gran_level='h', agg_func=agg_funcs).dropna()
    metrics_hour = compare_with_linear_reg(hour, target, ax=axs[1], plot_subtitle="freq='h'")
    print("rule='h':")
    print(DataFrame(metrics_hour))

    # rule: d
    day: DataFrame = ts_aggregation_by(df, gran_level='d', agg_func=agg_funcs).dropna()
    metrics_day = compare_with_linear_reg(day, target, ax=axs[2], plot_subtitle="freq='d'")
    print("rule='d':")
    print(DataFrame(metrics_day))

    fig.tight_layout()

    if savefig:
        fig.savefig(f'temp/{target}_aggregation_analysis_vs_lr.png')
        results = DataFrame({
            'no-agg': metrics_raw['test'],
            'h': metrics_hour['test'],
            'd': metrics_day['test']
        })
        results.transpose().to_csv(f'temp/{target}_aggregation_analysis_vs_lr.txt')


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
