from pandas import DataFrame, Series
from numpy import arange
from models import LinearRegression
from pipelines.tasks import prepare, save_report
# from pipelines.tasks.prepare import prepare
# from pipelines.tasks.report import save_report
from utils import get_options_with_default

DEFAULT_LINEAR_REGRESSION_OPTS = {
    'training_pct': 0.8,
    'smoothing': False
}


def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_LINEAR_REGRESSION_OPTS)

    print('\n-- Linear Regression --') 
    print(f'target: {target}')

    # For Linear Regression we only need to keep the target variable.
    df = df[[target]].copy()

    # Prepare dataset
    train, test = prepare(df)
    train = train[target]
    test = test[target]

    train_x = arange(len(train)).reshape(-1, 1)
    train_y = train.to_numpy()
    test_x = arange(len(train), len(df)).reshape(-1, 1)
    test_y = test.to_numpy()

    # Model data
    model = LinearRegression()
    model.fit(train_x, train_y)
    
    prd_trn: Series = Series(model.predict(train_x), index=train.index)
    prd_tst: Series = Series(model.predict(test_x), index=test.index)

    print(f'Intercept: {model.intercept_:.2f}')
    print(f'Coef: {model.coef_[0]:.2f}')

    # Save results to `path`
    if path:
        save_report(
            'linear-regression',
            target,
            train,
            test,
            prd_trn,
            prd_tst,
            observations=[f'Intercept: {model.intercept_:.2f}', f'Coef: {model.coef_[0]:.2f}'],
            path=path
        )

    return train, test, prd_trn, prd_tst
