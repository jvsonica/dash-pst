from numpy import array
from pandas import DataFrame, Series
from matplotlib.pyplot import Figure, figure, subplots
from matplotlib.gridspec import GridSpec
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


def create_autocorrelation_lagged_fig(series:Series, max_lag:int = 20, delta:int = 10):
    """Compare series to its lag on different intervals. Intervals can be defined with
    `max_lag` and `delta`. For instance, with a `max_lag` of 20 and `delta` of 10, we
    will be generating the lagged series for values 0, 10 and 20.

    Args:
        series (Series): Series to be analyzed.
        max_lag (int): Maximum series lag calculalated and plotted. Defaults to 20.
        delta (int): Lag step applied on each iteration. Defaults to 10.
    """
    def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
        lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
        for i in range(delta, max_lag + 1, delta):
            lagged_series[f"lag {i}"] = series.shift(i)
        return lagged_series


    lags = get_lagged_series(series, max_lag, delta)
    fig = figure(figsize=(3 * HEIGHT, HEIGHT))
    fig.suptitle(f"{series.name} {', '.join(lags.keys())} ({series.index[0]} until {series.index[-1]})")
    plot_multiline_chart(series.index.to_list(), lags, xlabel=series.index, ylabel=series.name)
    return fig

def create_autocorrelation_study_fig(series: Series, max_lag:int = 10, delta:int = 1):
    """Comparing the original series with each lagged series. Lag intervals can be
    defined with `max_lag` and `delta`. For instance, with a `max_lag` of 20 and `delta`
    of 10, we will be generating the lagged series for values 0, 10 and 20.

    Args:
        series (Series): Series to be analyzed.
        max_lag (int): Maximum series lag calculalated and plotted. Defaults to 10.
        delta (int): Lag step applied on each iteration. Defaults to 1.
    """
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        ax.scatter(series.shift(lag).tolist(), series_values)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series, maxlags=max_lag)
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    return fig


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
    print('\n-- Distribution --')
    series: Series = df[target]

    fig_five_number_summ = create_five_number_summary_fig(series)
    fig_variable_distribution = create_variable_distribution_fig(series)
    fig_autocorrelation_lags = create_autocorrelation_lagged_fig(series, max_lag=90, delta=30)
    fig_autocorrelation_study = create_autocorrelation_study_fig(series, max_lag = 10, delta = 1)

    # LA: When comparing with lagged, for better visibility, I am using the aggregated hourly series.
    # could using different max_lag/delta help out?
    ss_hourly = ts_aggregation_by(series, gran_level="D", agg_func='mean')
    fig_autocorrelation_study_hourly = create_autocorrelation_study_fig(ss_hourly, max_lag = 10, delta = 1)

    
    if savefig:
        fig_five_number_summ.savefig(f'temp/{target}_distribution-5-num-summary.png')
        fig_variable_distribution.savefig(f'temp/{target}_distribution-variable-histograms.png')
        fig_autocorrelation_lags.savefig(f'temp/{target}_distribution-autocorrelation-lags.png')
        fig_autocorrelation_study.savefig(f'temp/{target}_distribution-autocorrelation-study.png')
        fig_autocorrelation_study_hourly.savefig(f'temp/{target}_distribution-autocorrelation-study-hourly.png')
        print(f'saved temp/{target}_distribution-5-num-summary.png')
        print(f'saved temp/{target}_distribution-variable-histograms.png')
        print(f'saved temp/{target}_distribution-autocorrelation-lags.png')
        print(f'saved temp/{target}_distribution-autocorrelation-study.png')
        print(f'saved temp/{target}_distribution-autocorrelation-study-hourly.png')
    else:
        fig_five_number_summ.show()
        fig_variable_distribution.show()
        fig_autocorrelation_lags.show()
        fig_autocorrelation_study.show()
        fig_autocorrelation_study_hourly.show()
