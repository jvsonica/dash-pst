from pandas import DataFrame, Series
from pipelines.tasks.prepare import prepare
from models import RollingMeanRegressor
from utils import get_options_with_default
from pipelines.tasks import prepare, save_report, evaluate
from dslabs import DELTA_IMPROVE, plot_line_chart, HEIGHT
from matplotlib.pyplot import figure


DEFAULT_ROLLING_MEAN_OPTIONS = {
    'training_pct': 0.8,
    'smoothing': False,
    'optimize_for': 'R2'  # MAPE
}


def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_ROLLING_MEAN_OPTIONS)

    print('\n-- Rolling Mean --')
    print(f'target: {target}')

    # Prepare dataset
    train, test = prepare(df, options)
    train = train[target]
    test = test[target]

    # Model data
    win_size = (2, 4, 6, 12, 24, 36, 48, 48+12, 48+24, 96, 96+48, 192, 192 + 48 + 48, 192+48+192)
    metric = options['optimize_for']
    best_result: dict = {"metric": metric, "params": (), "predicted_test": None, "predicted_train": None, "perf": -10000}

    yvalues = []
    for w in win_size:
        pred = RollingMeanRegressor(win=w)
        pred.fit(train)
        prd_train = pred.predict(train)
        prd_test = pred.predict(test)

        metrics: dict = evaluate(train, test, prd_train, prd_test)
        eval: float = metrics['test'][metric]
        if eval > best_result["perf"] and abs(eval - best_result["perf"]) > DELTA_IMPROVE:
            best_result["perf"] = eval
            best_result["params"] = (w,)
            best_result["predicted_test"] = prd_test
            best_result["predicted_train"] = prd_train
            best_result["model"] = pred
        yvalues.append(eval)
        print(f"w={w} got {eval}")

    print(f"Best model using win={best_result['params'][0]} ({win_size})")

    # Save results to `path`
    if path:
        save_report(
            f'rolling-mean-{metric}-win={best_result["params"][0]}',
            target,
            train,
            test,
            best_result['predicted_train'],
            best_result['predicted_test'],
            observations=[f"best model using win={best_result['params'][0]} ({win_size})"],
            path=path
        )
    
    fig = figure(figsize=(3 * HEIGHT, HEIGHT))
    ax = plot_line_chart(
        win_size, yvalues, title=f"Rolling Mean ({metric})", xlabel="window size", ylabel=metric, percentage=True
    )
    ax.set_ylim(-3, 1)
    fig.tight_layout()
    fig.savefig(f"{path}/rolling-mean-parameter-tuning-for-{metric}.png")
