from pandas import DataFrame, Series
from matplotlib.pyplot import subplots, plot, legend, figure, Figure
from matplotlib.axes import Axes
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from dslabs import HEIGHT, set_chart_labels, ts_aggregation_by, plot_line_chart
from utils import log_execution_time

def plot_components(
    series: Series
) -> list[Axes]:
    """Perform and plot seasonal decomposition using moving averages.
    This outputs a figure with plots for:
    - original: the original data in `series`.
    - trend: long-term pattern in the data, ignoring short-term fluctuations.
    - seasonality: repeating patterns or cycles with a fixed frequency over time.
    - noise: random or irregular fluctuations.

    Args:
        series (Series): Series to be analyzed.

    Returns:
        Figure: Figure with series components decomposed.
    """
    decomposition: DecomposeResult = seasonal_decompose(series, model="add")
    components: dict = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    rows: int = len(components)
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{series.name} hourly")
    i: int = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=series.index.name, ylabel=series.name)
        axs[i].plot(components[key])
        i += 1
    fig.tight_layout()
    return fig


def plot_with_mean(series:Series):
    """Generate a figure with the original series and its mean.

    Args:
        series (Series): Series to be analyzed.

    Returns:
        Figure: Figure with series and its mean.
    """
    fig = figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        series.index.to_list(),
        series.to_list(),
        xlabel=series.index.name,
        ylabel=series.name,
        title=f"{series.name} stationary study",
        name="original",
    )
    n: int = len(series)
    plot(series.index, [series.mean()] * n, "r-", label="mean")
    legend()
    fig.tight_layout()
    return fig


def plot_with_binned_mean(series:Series, bins:int = 10):
    """Generate a figure with the original series and its rolling mean.

    Args:
        series (Series): series to be analyzed 
        bins (int, optional): Amount of bins to divide data in. Defaults to 10.

    Returns:
        Figure: Figure with binned mean
    """
    n: int = len(series)
    mean_line: list[float] = []

    for i in range(bins):
        segment: Series = series[i * n // bins : (i + 1) * n // bins]
        mean_value: list[float] = [segment.mean()] * (n // bins)
        mean_line += mean_value
    mean_line += [mean_line[-1]] * (n - len(mean_line))

    fig = figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        series.index.to_list(),
        series.to_list(),
        xlabel=series.index.name,
        ylabel=series.name,
        title=f"{series.name} stationary study",
        name="original",
        show_stdev=True,
    )
    n: int = len(series)
    plot(series.index, mean_line, "r-", label="mean")
    legend()
    fig.tight_layout()
    return fig


@log_execution_time
def eval_stationarity_adf(series: Series):
    """Evaluate if `series` is stationary using the Augmented Dickey-Fuller test.
    In this test, we get a p-value that when:
    - p-value <= 0.05 : the series is stationary, meaning its values do not depend on time;
    - p-value > 0.05 : the series is non-stationary, meaning it shows a time-dependent structure.
    Args:
        series (Series): Series to be analyzed.
    """
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")

    stationarity = [
        f"{series.name}",
        f"ADF Statistic: {result[0]:.3f}",
        f"p-value: {result[1]:.3f}",
        f"Critical values:",
        *[f"\t{key}: {value:.3f}" for key, value in result[4].items()],
        f"The series {('is' if result[1] <= 0.05 else 'is not')} stationary"
    ]
    with open(f"temp/{series.name}_stationarity.txt", "w") as f:
        f.write("\n".join(stationarity))
    print(f"saved temp/{series.name}_stationarity.txt")

    return result[1] <= 0.05

def analyze(df: DataFrame, target: str, savefig=True):
    """Analyze stationarity of a series.

    Args:
        df (DataFrame): DataFrame to analyze
        target (str): Column name to analyze
        savefig (bool, optional): Save figures to files. Defaults to True.
    """
    print('\n-- Stationarity --')
    series = df[target]
    ss_hourly: Series = ts_aggregation_by(series, gran_level="h", agg_func='mean')
    ss_hourly = ss_hourly.asfreq('h')
    ss_hourly = ss_hourly.ffill()

    fig_components = plot_components(ss_hourly)
    fig_mean = plot_with_mean(ss_hourly)
    fig_binned_mean = plot_with_binned_mean(ss_hourly)

    is_stationary= eval_stationarity_adf(ss_hourly)
    print(f"The series {('is' if is_stationary else 'is not')} stationary")

    if savefig:
        fig_components.savefig(f'temp/{target}_stationary-components.png')
        fig_mean.savefig(f'temp/{target}_stationary-mean.png')
        fig_binned_mean.savefig(f'temp/{target}_stationary-binned-mean.png')
        print(f'saved temp/{target}_stationary-components.png')
        print(f'saved temp/{target}_stationary-mean.png')
        print(f'saved temp/{target}_stationary-binned-mean.png')
    else:
        fig_components.show()
        fig_mean.show()
        fig_binned_mean.show()
