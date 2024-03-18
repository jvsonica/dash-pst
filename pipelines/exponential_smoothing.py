import os
from pandas import read_csv, DataFrame, Series
from pipelines.tasks.prepare import prepare
from utils import get_options_with_default
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from pipelines.tasks import prepare, save_report, evaluate
from dslabs import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart, HEIGHT
from matplotlib.pyplot import figure

# from dslabs_functions import series_train_test_split, HEIGHT

DEFAULT_EXPONENTIAL_SMOOTHING_OPTIONS = {
    'training_pct': 0.8,
    'smoothing': False,
    'optimize_for': 'R2'  # MAPE
}


def exponential_smoothing_study(train: Series, test: Series, measure: str = "R2"):
    alpha_values = [i / 10 for i in range(1, 10)]
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Exponential Smoothing", "metric": measure, "params": ()}
    best_performance: float = -100000

    yvalues = []
    for alpha in alpha_values:
        tool = ExponentialSmoothing(train)
        model = tool.fit(smoothing_level=alpha, optimized=False)
        prd_tst = model.forecast(steps=len(test))

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (alpha,)
            best_model = model
        yvalues.append(eval)

    print(f"Exponential Smoothing best with alpha={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    ax = plot_line_chart(
        alpha_values,
        yvalues,
        title=f"Exponential Smoothing ({measure})",
        xlabel="alpha",
        ylabel=measure,
        percentage=flag,
    )
    ax.set_ylim(-3, 1)

    return best_model, best_params


def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_EXPONENTIAL_SMOOTHING_OPTIONS)

    # Create target dir if it doesn't exist
    os.makedirs(path, exist_ok=True)

    metric = options['optimize_for']

    print('\n-- EXponential Smoothing --')
    print(f'target: {target}')

    # Prepare dataset
    train, test = prepare(df, options)
    train = train[target]
    test = test[target]

    fig = figure(figsize=(3 * HEIGHT, HEIGHT))
    best_model, best_params = exponential_smoothing_study(train, test, measure=metric)
    fig.tight_layout()
    fig.savefig(f"{path}/exponential_smoothing_{metric}_study.png")

    prd_trn = best_model.predict(start=0, end=len(train) - 1)
    prd_tst = best_model.forecast(steps=len(test))

        # Save results to `path`
    if path:
        save_report(
            f'exponential-smoothing-{metric}',
            target,
            train,
            test,
            prd_trn,
            prd_tst,
            observations=[f"best model using win={best_params['params']})"],
            path=path
        )

