import os
from pandas import DataFrame, Series
from transformation import smoothing
from matplotlib.pyplot import savefig
from dslabs import (
    series_train_test_split,
    plot_forecasting_series,
    plot_forecasting_eval,
)
from models import SimpleAvgRegressor, evaluate
from pipelines.prepare import aggregation_func_by_col
from utils import get_options_with_default


DEFAULT_SIMPLE_AVERAGE_OPTIONS = {
    'training_pct': 0.8,
    'smoothing': { 'window': 50 }
}


def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_SIMPLE_AVERAGE_OPTIONS)

    # Create target dir if it doesn't exist
    os.makedirs(path, exist_ok=True)

    print('\n-- Simple Average --')
    print(f'target: {target}')

    # For Simple Average we only need to keep the target variable.
    df = df[[target]].copy()

    # Test/Train Split
    train, test = series_train_test_split(data=df, trn_pct=options['training_pct'])

    # Smooth on training set
    if 'smoothing' in options and options['smoothing']:
        agg_func = aggregation_func_by_col[target]
        train = smoothing.run(train, window=options['smoothing']['window'], agg_func=agg_func)
        train = train[~train.isna()].copy()

    # Model data
    fr_mod = SimpleAvgRegressor()
    fr_mod.fit(train)
    prd_trn: Series = fr_mod.predict(train)
    prd_tst: Series = fr_mod.predict(test)

    print(f'Prediction: {prd_tst.iloc[0]:.2f}')
    results = evaluate(train, test, prd_trn, prd_tst)
    for metric, value in results['test'].items():
        print(f'{metric}:\t{value}')

    # Save results to `path`
    if path:
        plot_forecasting_eval(
            train, test, prd_trn, prd_tst, title=f"{target} - Simple Average"
        )
        savefig(f"{path}/simple-average-{target}-eval.png")

        plot_forecasting_series(
            train,
            test,
            prd_tst,
            title=f"{target} - Simple Average",
            xlabel=train.index.name,
            ylabel=target,
        )
        savefig(f"{path}/simple-average-{target}-forecast.png")

        with open(f'{path}/simple-average-{target}-run.txt', 'w') as f:
            f.write(f'target: {target}\n')
            f.write(f'start: {df.index[0]}\n')
            f.write(f'end: {df.index[-1]}\n')
            f.write('test\n')
            f.write('\n'.join([f'{metric}: {value}' for (metric, value) in results['test'].items()]))
            # f.write('\n')
            # f.write('Train\n')
            # f.write('\n'.join([f'{metric}:\t{value}' for (metric, value) in results['train'].items()]))
