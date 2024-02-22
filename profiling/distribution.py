from numpy import array
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show, subplots, Figure
from dslabs import HEIGHT, plot_multiline_chart, ts_aggregation_by, set_chart_labels

def create_five_number_summary_fig(series: Series):
    """Generate a box plot and describe the Series passed as parameter.
    This helps us get a grasp of the distribution of the values of the
    Series. 

    Args:
        series (Series): Series to be analyzed.
    """
    # LA: for temperature variables, there's no point in trying the sum aggfunc
    ss_hourly: Series = ts_aggregation_by(series, gran_level="h", agg_func='mean')
    ss_weekly: Series = ts_aggregation_by(series, gran_level="W", agg_func='mean')

    fig: Figure
    axs: array
    fig, axs = subplots(2, 2, figsize=(2 * HEIGHT, HEIGHT))
    set_chart_labels(axs[0, 0], title="HOURLY")
    axs[0, 0].boxplot(ss_hourly.values)
    set_chart_labels(axs[0, 1], title="WEEKLY")
    axs[0, 1].boxplot(ss_weekly)

    axs[1, 0].grid(False)
    axs[1, 0].set_axis_off()
    axs[1, 0].text(0.2, 0, str(ss_hourly.describe()), fontsize="small")

    axs[1, 1].grid(False)
    axs[1, 1].set_axis_off()
    axs[1, 1].text(0.2, 0, str(ss_weekly.describe()), fontsize="small")

    return fig


def create_variable_distribution_fig(series: Series):
    """Generate histogram plots over different aggregation periods to show us the
    variable distribution with different granularities.

    Args:
        series (Series): Series to be analyzed.
    """
    ss_daily: Series = ts_aggregation_by(series, gran_level="D", agg_func='mean')
    ss_weekly: Series = ts_aggregation_by(series, gran_level="W", agg_func='mean')
    ss_monthly: Series = ts_aggregation_by(series, gran_level="M", agg_func='mean')
    ss_quarterly: Series = ts_aggregation_by(series, gran_level="Q", agg_func='mean')

    grans: list[Series] = [series, ss_daily, ss_weekly, ss_monthly, ss_quarterly]
    gran_names: list[str] = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly"]

    fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
    fig.suptitle(f"{series.name}")
    for i in range(len(grans)):
        set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=series.name, ylabel="Nr records")
        axs[i].hist(grans[i].values)
    return fig


def create_autocorrelation_fig(series: Series):
    """_summary_

    Args:
        series (Series): Series to be analyzed.
    """
    # def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    #     lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    #     for i in range(delta, max_lag + 1, delta):
    #         lagged_series[f"lag {i}"] = series.shift(i)
    #     return lagged_series


    # figure(figsize=(3 * HEIGHT, HEIGHT))
    # lags = get_lagged_series(series, 20, 10)
    # plot_multiline_chart(series.index.to_list(), lags, xlabel=series.index, ylabel=series.name)
    pass


def analyze(df: DataFrame, target: str, savefig = True):
    """Print and save distribution analysis of `target` in the context of the timeseries
    dataset in `df`.
    Ensure that your dataframe index is a DatetimeIndex. This means that 
    `isinstance(df.index, pd.DatetimeIndex)` should be truthy.

    This invokes routines that perform:
    - 5 number summary
    - Variable distribution (histogram)

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing distribution
    """
    series: Series = df[target]

    fig_five_number_summ = create_five_number_summary_fig(series)
    fig_variable_distribution = create_variable_distribution_fig(series)
    
    if savefig:
        fig_five_number_summ.savefig(f'temp/{target}_distribution-5-num-summary.png')
        fig_variable_distribution.savefig(f'temp/{target}_distribution-variable-histograms.png')
    else:
        fig_five_number_summ.show()
        fig_variable_distribution.show()
