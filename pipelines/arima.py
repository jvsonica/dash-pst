import os
import traceback
import time
from pandas import DataFrame, Series, DatetimeIndex
from statsmodels.tsa.arima.model import ARIMA
from dslabs import HEIGHT, DELTA_IMPROVE, subplots, plot_multiline_chart
from utils import get_options_with_default, log_execution_time
from pipelines.tasks import prepare, save_report, evaluate

"""

The best parameterisation of ARIMA (p=2, d=0 and q=7) provided the best results so far, with an R2 of -0.11. 
Applying differentiation (d=1 or d=2) yielded much worse results. 
For the exogenous variable, a charging session timer was used. 
In the context of the problem, the temperature is expected to rise with longer charging sessions.

"""


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
    fig.tight_layout()
    fig.savefig(f'{path}/arima-{target}-diagnostics.png')


@log_execution_time
def find_best_parameters(train, test, optimize_for='R2', exogenous=None, path='./temp'):
    metric = optimize_for
    show_by_perc = metric == "R2" or metric == "MAPE"
    best_result: dict = {"metric": metric, "params": (), "predicted_test": None, "predicted_train": None, "perf": -10000}

    d_values = [0, 1, 2]
    p_params = [1, 2, 3, 5, 7, 10]
    q_params = [1, 3, 5, 7]

    # d_values = [0,1,2]
    # p_params = [1]
    # q_params = [1]

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for d in d_values:
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                start = time.time()
                if exogenous is not None:
                    arima = ARIMA(train, exog=exogenous['train'], order=(p, d, q), missing='none')
                    model = arima.fit()
                    prd_test = model.forecast(steps=len(test), exog=exogenous['test'], signal_only=False)
                else:
                    arima = ARIMA(train, order=(p, d, q), missing='none')
                    model = arima.fit()
                    prd_test = model.forecast(steps=len(test), signal_only=False)

                metrics: dict = evaluate(test=test, predicted_test=prd_test)
                eval: float = metrics['test'][metric]
                print(f'({p},{d},{q}): {eval}. took {time.time() - start}')

                if eval > best_result["perf"] and abs(eval - best_result["perf"]) > DELTA_IMPROVE:
                    best_result["perf"] = eval
                    best_result["params"] = (p, d, q)
                    best_result["predicted_test"] = prd_test
                    best_result["model"] = model
                yvalues.append(eval)
            values[q] = yvalues

        ax = plot_multiline_chart(
            p_params, values, ax=axs[d], title=f"ARIMA d={d} ({metric})", xlabel="p", ylabel=metric, percentage=show_by_perc
        )
        ax.set_ylim(-3.5, 1)

    print(
        f"ARIMA best results achieved with (p,d,q)=({best_result['params'][0]:.0f}, {best_result['params'][1]:.0f}, {best_result['params'][2]:.0f}) ==> measure={best_result['perf']:.2f}"
    )

    fig.savefig(f"{path}/arima-parameter-tuning-by-d-for-{metric}.png")

    return best_result


def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_ARIMA_OPT_REGRESSOR_OPTIONS)

    # Create target dir if it doesn't exist
    os.makedirs(path, exist_ok=True)

    print(f"\n-- ARIMA {'EXOG' if 'exogenous' in options else ''} --")
    print(f'target: {target}')

    # Prepare dataset
    train, test = prepare(df, options)
    target_train = train[target]
    target_test = test[target]

    # diagnostics(train, test, target='system_battery_max_temperature', path=path)

    exogenous = None
    if 'exogenous' in options and len(options['exogenous']) > 0:
        exogenous = [{ 'train': train[exog], 'test': test[exog] } for exog in options['exogenous']]
        best_result = find_best_parameters(target_train, target_test, exogenous=exogenous[0], optimize_for=options['optimize_for'], path=path)
        prd_train = best_result['model'].predict(start=0, end=len(target_train) - 1)
        prd_test = best_result['model'].forecast(steps=len(target_test), exog=target_test)

    else:
        best_result = find_best_parameters(target_train, target_test, optimize_for=options['optimize_for'], path=path)
        prd_train = best_result['model'].predict(start=0, end=len(target_train) - 1)
        prd_test = best_result['model'].forecast(steps=len(target_test))

    # Save results to `path`
    if path:
        save_report(
            'arima',
            target,
            target_train,
            target_test,
            prd_train,
            prd_test,
            observations=[
                f"optimized_for={best_result['metric']}",
                f"(p,d,q)={best_result['params']}"
            ],
            path=path
        )
