import os
import itertools
from pandas import DataFrame, Series, DatetimeIndex
from statsmodels.tsa.arima.model import ARIMA
from dslabs import HEIGHT, DELTA_IMPROVE, subplots, plot_multiline_chart
from utils import get_options_with_default, log_execution_time
from pipelines import prepare, evaluate, save_report


DEFAULT_ARIMA_OPT_REGRESSOR_OPTIONS = {
    'training_pct': 0.8,
    'smoothing': False,
    'optimize_for': 'R2'  # R2 or MAPE
}

def diagnostics(train, test, target, path):
    predictor = ARIMA(train, order=(3, 1, 2))
    model = predictor.fit()

    print(model.summary())

    fig = model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
    fig.savefig(f'{path}/arima-{target}-diagnostics.png', 'w')


@log_execution_time
def find_best_parameters(train, test, optimize_for='R2', path='./temp'):
    metric = optimize_for
    show_by_perc = metric == "R2" or metric == "MAPE"
    best_result: dict = {"metric": metric, "params": (), "predicted_test": None, "predicted_train": None, "perf": -10000}

    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for d in d_values:
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                print(d, p, q)
                arima = ARIMA(train, order=(p, d, q))
                model = arima.fit()
                prd_test = model.forecast(steps=len(test), signal_only=False)
                metrics: dict = evaluate(test=test, predicted_test=prd_test)
                eval: float = metrics['test'][metric]
        
                if eval > best_result["perf"] and abs(eval - best_result["perf"]) > DELTA_IMPROVE:
                    best_result["perf"] = eval
                    best_result["params"] = (p, d, q)
                    best_result["predicted_test"] = prd_test
                    best_result["model"] = model
                yvalues.append(eval)
            values[q] = yvalues

        plot_multiline_chart(
            p_params, values, ax=axs[d], title=f"ARIMA d={d} ({metric})", xlabel="p", ylabel=metric, percentage=show_by_perc
        )

    print(
        f"ARIMA best results achieved with (p,d,q)=({best_result['params'][0]:.0f}, {best_result['params'][1]:.0f}, {best_result['params'][2]:.0f}) ==> measure={best_result['perf']:.2f}"
    )

    fig.savefig(f"temp/arima-parameter-tuning-for-{metric}.png")

    return best_result


def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_ARIMA_OPT_REGRESSOR_OPTIONS)

    # Create target dir if it doesn't exist
    os.makedirs(path, exist_ok=True)

    print('\n-- ARIMA --')
    print(f'target: {target}')

    # Prepare dataset
    df.index = DatetimeIndex(df.index).to_period('H')
    train, test = prepare(df, options)

    # diagnostics(train, test, target='system_battery_max_temperature', path=path)

    best_result = find_best_parameters(train, test, optimize_for=options['optimize_for'], path=path)

    print(best_result)

    prd_train = best_result['model'].predict(start=0, end=len(train) - 1)
    prd_test = best_result['model'].forecast(steps=len(test))
    metrics = evaluate(train, test, prd_train, prd_test)

    # Save results to `path`
    if path:
        save_report(
            'arima',
            target,
            train,
            test,
            prd_train,
            prd_test,
            path=path
        )
        pass


