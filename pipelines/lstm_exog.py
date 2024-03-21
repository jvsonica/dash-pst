import torch
import time
import os
import pandas as pd
from pandas import DataFrame, Series, read_csv
from pipelines.tasks.prepare import prepare
from models import DS_LSTM_Exog, prepare_dataset_for_lstm_exog
from utils import get_options_with_default
from pipelines.tasks import prepare, save_report, evaluate
from dslabs import DELTA_IMPROVE, plot_line_chart, HEIGHT
from matplotlib.pyplot import figure, subplots
from dslabs import DELTA_IMPROVE, plot_forecasting_series, plot_line_chart, HEIGHT, FORECAST_MEASURES, plot_multiline_chart
from copy import deepcopy
from utils import log_execution_time
from transformation import smoothing
import matplotlib.pyplot as plt

DEFAULT_LSTM_OPTIONS = {
    'training_pct': 0.8,
    'smoothing': False,
    'optimize_for': 'R2'  # or MAPE
}

def run(df: DataFrame, target: str, options: dict|None= None, path='temp/'):
    # Resolve options taking into consideration defaults
    options = get_options_with_default(options, DEFAULT_LSTM_OPTIONS)

    # Create target dir if it doesn't exist
    os.makedirs(path, exist_ok=True)

    print('\n-- LSTM --')
    print(f'target: {target}')

    # Prepare dataset
    # df = df.copy()
    # df_smooth = smoothing.run(df, window).iloc[window-1:]
    series = df[[target]]
    train_size = int(len(series) * options['training_pct'])
    train, test = series[:train_size], series[train_size:] 
    # train, test = prepare(df, options)
    if 'smoothing' in options and options['smoothing']:
        window = 12
        train = smoothing.run(train, window).iloc[window-1:]

    np_train = train.values.astype('float32')
    np_test = test.values.astype('float32')

    cols = ['system_battery_max_temperature', 'system_grid_session_duration', 'system_battery_soc']
    series = df[cols].values.astype("float32")
    train_size = int(len(series) * 0.80)
    train, test = series[:train_size], series[train_size:]

    np_train = train.copy()
    np_test = test[:,0].reshape(len(test), 1)

    print('train', len(np_train), type(np_train), np_train.shape)
    print('test', len(np_test), type(np_test), np_test.shape)

    metric = options['optimize_for']

    # model = DS_LSTM_Exog(train, input_size=len(cols), hidden_size=50, num_layers=1)
    # loss = model.fit()
    # tensor_x, tensor_y = prepare_dataset_for_lstm_exog(np_test, seq_length=4)
    # model.predict(tensor_x)
    # print(loss)

    best_model, best_params = lstm_study(np_train, np_test, nr_episodes=1000, measure=metric, path=path)

    params = best_params["params"]
    best_length = params[0]
    trnX, trnY = prepare_dataset_for_lstm_exog(np_train, seq_length=best_length)
    tstX, tstY = prepare_dataset_for_lstm_exog(np_test, seq_length=best_length)

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)

    # Prepare sets for plot-compatible formats
    prd_trn = Series(prd_trn.numpy().ravel(), index=df[target][best_length:train_size].index)
    prd_tst = Series(prd_tst.numpy().ravel(), index=df[target][train_size+best_length:].index)
    

    # Save results to `path`
    if path:
        save_report(
            f'lstm-{metric}',
            target,
            df[target][best_length:train_size],
            df[target][train_size+best_length:],
            prd_trn,
            prd_tst,
            observations=[f"{params}"],
            path=path
        )


@log_execution_time
def lstm_study(train, test, nr_episodes: int = 1000, measure: str = "R2", path='temp/'):
    sequence_size = [6, 8, 12]
    nr_hidden_units = [2, 10, 25]

    step: int = nr_episodes // 10
    # step: int = 10
    episodes = [1, 10, 20, 30, 40] + list(range(0, nr_episodes + 1, step))[1:]
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "LSTM", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(sequence_size), figsize=(len(sequence_size) * HEIGHT, HEIGHT))
    # ax_test.plot(test.ravel(), label='test')

    for i in range(len(sequence_size)):
        length = sequence_size[i]
        tstX, tstY = prepare_dataset_for_lstm_exog(test, seq_length=length)

        values = {}
        for hidden in nr_hidden_units:
            yvalues = []
            model = DS_LSTM_Exog(train, input_size=train.shape[1], hidden_size=hidden)
            for n in range(0, nr_episodes + 1):
                model.fit()

                if n in episodes:
                    model.eval()  # Set model to evaluation mode
                    with torch.no_grad(): 
                        to_test = deepcopy(model)
                        prd_tst = to_test.predict(tstX)
                    eval: float = FORECAST_MEASURES[measure](test[length:], prd_tst)
                    print(f"seq length={length} hidden_units={hidden} nr_episodes={n}", eval)
                    if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                        best_performance: float = eval
                        best_params["params"] = (length, hidden, n)
                        best_model = deepcopy(model)
                    yvalues.append(eval)
            values[hidden] = yvalues
        plot_multiline_chart(
            episodes,
            values,
            ax=axs[i],
            title=f"LSTM seq length={length} ({measure})",
            xlabel="nr episodes",
            ylabel=measure,
            percentage=flag,
        )
    print(
        f'LSTM best results achieved with length={best_params["params"][0]} hidden_units={best_params["params"][1]} and nr_episodes={best_params["params"][2]}) ==> measure={best_performance:.2f}'
    )
    # mby savefig
    fig.savefig(f'{path}/lstm-study.png')
    return best_model, best_params

