import os
from pandas import DataFrame, Series
from models import PersistenceOptimistRegressor
from pipelines import prepare
from pipelines import save_report
from utils import get_options_with_default


DEFAULT_PERSISTENCE_OPT_REGRESSOR_OPTIONS = {
    'training_pct': 0.8,
    'smoothing': { 'window': 50 },
}


def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_PERSISTENCE_OPT_REGRESSOR_OPTIONS)

    # Create target dir if it doesn't exist
    os.makedirs(path, exist_ok=True)

    print('\n-- Persistence Optimistic Regressor --')
    print(f'target: {target}')

    # Prepare dataset
    train, test = prepare(df, options)

    # Model data
    fr_mod = PersistenceOptimistRegressor()
    fr_mod.fit(train)
    prd_trn: Series = fr_mod.predict(train)
    prd_tst: Series = fr_mod.predict(test)

    # Save results to `path`
    if path:
        save_report(
            'persistence-optimistic',
            target,
            train,
            test,
            prd_trn,
            prd_tst,
            path=path
        )
