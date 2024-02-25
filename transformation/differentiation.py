from pandas import DataFrame, Series
from dslabs import plot_line_chart, plot_ts_multivariate_chart, HEIGHT
from matplotlib.pyplot import figure


def run(df: DataFrame, target: str, savefig=True):
    """Plot differentiation applied to df.

    Args:
        df (DataFrame): Dataframe with time series information. 
        target (str): Column name of `df` from which we are analyzing granularity.
        savefig (bool, optional): Save generated figures to files. Defaults to True.

    Returns
        DataFrame: Data with differentiation applied to all variables.
    """
    series: Series = df[target]
    ss_diff: Series = series.diff()
    fig_target_diff = figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        ss_diff.index.to_list(),
        ss_diff.to_list(),
        title=f"{target} differentiation",
        xlabel=series.index.name,
        ylabel=target,
    )
    fig_target_diff.tight_layout()

    diff_df: DataFrame = df.diff()
    axs_full_diff = plot_ts_multivariate_chart(diff_df, title=f"{target} - after first differentiation")
    fig_full_diff = axs_full_diff[0].get_figure()
    fig_full_diff.tight_layout()

    if savefig:
        fig_target_diff.savefig(f'temp/{target}_differentiation-target.png')
        fig_full_diff.savefig(f'temp/{target}_differentiation-mv.png')
    else:
        fig_target_diff.show()
        fig_full_diff.show()

    return diff_df
