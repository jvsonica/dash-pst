from pandas import DataFrame, Series
from dslabs import plot_forecasting_series_on_ax
from matplotlib.pyplot import subplots
from pipelines import linear_regression
from pipelines.tasks.evaluate import evaluate

def compare_with_linear_reg(df, target, ax=None, plot_subtitle=""):
    train, test, predicted_train, predicted_test = linear_regression.run(df, target, path=None)
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


def analyze(df: DataFrame, target: str, plot_title='Differentiation Analysis', savefig=True, ):
    """Plot differentiation applied to df.

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing granularity.
        savefig (bool, optional): Save generated figures to files. Defaults to True.
    """
    # Perform differentiation
    diff_1 = df[[target]].diff().dropna()
    diff_2 = diff_1[[target]].diff().dropna()

    # Setup result plot
    fig, axs = subplots(3, 1, figsize=(16, 3 * 1.5))
    fig.suptitle(plot_title)

    # No diff
    metrics_raw = compare_with_linear_reg(df, target, ax=axs[0], plot_subtitle="no differentiation")
    print('No differentiation:')
    print(DataFrame(metrics_raw))

    # d=1
    metrics_d1 = compare_with_linear_reg(diff_1, target, ax=axs[1], plot_subtitle="d=1")
    print('d=1:')
    print(DataFrame(metrics_d1))

    # d=2
    metrics_d2 = compare_with_linear_reg(diff_2, target, ax=axs[2], plot_subtitle="d=2")
    print('d=2:')
    print(DataFrame(metrics_d2))

    fig.tight_layout()

    if savefig:
        fig.savefig(f'temp/differentiation_analysis_vs_lr_{target}.png')
        results = DataFrame({
            'no-diff': metrics_raw['test'],
            'd=1': metrics_d1['test'],
            'd=2': metrics_d2['test']
        })
        results.transpose().to_csv(f'temp/differentiation_analysis_vs_lr_{target}.txt')


def run(df: DataFrame):
    """Apply differentiation to entire dataframe.
    TODO: explain differentiation

    Args:
        df (DataFrame):

    Returns:
        DataFrame: 
    """
    return df.diff().dropna()
