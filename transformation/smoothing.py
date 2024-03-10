from pandas import Series, DataFrame
from dslabs import plot_forecasting_series_on_ax
from matplotlib.pyplot import Figure, Axes, subplots
from preprocess import aggregation_func_by_col
from pipelines import linear_regression
from pipelines.tasks.evaluate import evaluate, compare_with_linear_reg


def analyze(df: DataFrame, target: str, windows=None, plot_title='Smoothing Analysis', savefig=True):
    """Apply smoothing a series. Output a plot with multiple smoothing windows.
    """
    sizes: list[int] = windows
    if windows is None:
        sizes = [12, 24, 36, 48]

    series: Series = df[target]
    fig: Figure
    axs: list[Axes]
    metrics = {}
    plot_count = len(sizes) + 1
    fig, axs = subplots(plot_count, 1, figsize=(16, plot_count * 1.5))
    fig.suptitle(plot_title)

    # No diff
    metrics_raw = compare_with_linear_reg(df, target, ax=axs[0], plot_subtitle="no smoothing")
    metrics['no-smoothing'] = metrics_raw
    print('No smoothing:')
    print(DataFrame(metrics_raw))

    for i in range(1, len(sizes) + 1):
        window = sizes[i-1]
        df[f'window={window}'] = series.rolling(window=window).mean()
        iter_metrics = compare_with_linear_reg(
            df[[f'window={window}']].dropna(),
            f'window={window}',
            ax=axs[i],
            plot_subtitle=f'window={window}'
        )
        print(f'window={window}:')
        print(DataFrame(iter_metrics))
        metrics[f'window={window}'] = iter_metrics

    fig.tight_layout()

    if savefig:
        fig.savefig(f'temp/smoothing_mean_analysis_w_lr_{target}.png')
        results = DataFrame({ k: v['test'] for (k, v) in metrics.items() })
        results.transpose().to_csv(f'temp/smoothing_mean_analysis_w_lr_{target}.txt')


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
