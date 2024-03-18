from pandas import DataFrame, Series
from models import PersistenceRealistRegressor
from pipelines.tasks.prepare import prepare
from pipelines.tasks.report import save_report
from utils import get_options_with_default


DEFAULT_PERSISTENCE_REALISTIC_OPTIONS = {
    'training_pct': 0.8,
    'smoothing': False,
}


def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_PERSISTENCE_REALISTIC_OPTIONS)

    print('\n-- Persistence Realistic Regressor --')
    print(f'target: {target}')

    # For Simple Average we only need to keep the target variable.
    df = df[[target]].copy()

    # Prepare dataset
    train, test = prepare(df, options)
    train = train[target]
    test = test[target]

    # Model data
    fr_mod = PersistenceRealistRegressor()
    fr_mod.fit(train)
    prd_trn: Series = fr_mod.predict(train)
    prd_tst: Series = fr_mod.predict(test)

    print(f'Prediction: {prd_tst.iloc[0]:.2f}')

    # Save results to `path`
    if path:
        save_report(
            'persistence-long-term',
            target,
            train,
            test,
            prd_trn,
            prd_tst,
            path=path
        )
