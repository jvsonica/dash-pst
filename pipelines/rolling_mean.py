from pandas import DataFrame, Series
from pipelines.tasks.prepare import prepare
from models import RollingMeanRegressor
from pipelines import evaluate, save_report
from utils import get_options_with_default
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

    # For Rolling Mean we only need to keep the target variable.
    df = df[[target]].copy()

    # Prepare dataset
    train, test = prepare(df, options)

    # Model data
    win_size = (12, 24, 36, 48, 96, 192, 384, 768)
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

    print(f"Best model using win={best_result['params'][0]} ({win_size})")

    # Save results to `path`
    if path:
        save_report(
            f'rolling-mean-{metric}',
            target,
            train,
            test,
            best_result['predicted_train'],
            best_result['predicted_test'],
            observations=[f"best model using win={best_result['params'][0]} ({win_size})"],
            path=path
        )
    
    fig = figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        win_size, yvalues, title=f"Rolling Mean ({metric})", xlabel="window size", ylabel=metric, percentage=True
    )
    fig.tight_layout()
    fig.savefig(f"{path}/rolling-mean-parameter-tuning-for-{metric}.png")



"""
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                arima = ARIMA(train, order=(p, d, q))
                model = arima.fit()
                prd_tst = model.forecast(steps=len(test), signal_only=False)
                eval: float = FORECAST_MEASURES[measure](test, prd_tst)
                # print(f"ARIMA ({p}, {d}, {q})", eval)
                if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params["params"] = (p, d, q)
                    best_model = model
                yvalues.append(eval)
            values[q] = yvalues
        plot_multiline_chart(
            p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure, percentage=flag
        )
    print(
        f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
    )

"""