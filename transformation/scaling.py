from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from dslabs import plot_line_chart, HEIGHT
from matplotlib.pyplot import figure, show, subplots
from pipelines.tasks.evaluate import compare_with_linear_reg

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

    # Setup result plot
    no_plots = 2
    fig, axs = subplots(no_plots, 1, figsize=(16, no_plots * 1.5))
    fig.suptitle(f"Scaling Analysis on {target}")

    # No Scaling
    metrics_raw = compare_with_linear_reg(df, target, ax=axs[0], plot_subtitle="no scaling")
    print('No scaling:')
    print(metrics_raw)

    ss_scaled: Series = scaled[target]
    metrics_scaled = compare_with_linear_reg(
        scaled,
        f'system_battery_max_temperature',
        ax=axs[1],
        plot_subtitle=f'scaled'
    )
    print(metrics_scaled)
    
    fig.tight_layout()
    if savefig:
        fig.savefig(f'temp/smoothing_mean_analysis_w_lr_{target}.png')
        fig.savefig(f'temp/scaling_analysis_w_lr-{target}.png')

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
