import os
import pandas as pd
from preprocess import preprocess
from profiling import dimensionality, granularity, distribution, stationarity
from transformation import aggregation, differentiation, smoothing, scaling
from pipelines.tasks import transform
from pipelines import (
    simple_average,
    persistence_optimistic,
    persistence_realist,
    linear_regression,
    rolling_mean,
    exponential_smoothing,
    arima,
    lstm,
    lstm_exog
)


def load_data(dir):
    months = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            months.append(df)

    df = pd.concat(months, axis=0, ignore_index=True)
    df = preprocess(df, datetime_col="registered_at")
    return df

df = load_data("./data/1min")

# Using this when we have already loaded and preprocessed the df locally
# to save time on each execution.
# df = pd.read_pickle('./data/resampled-1hr.pkl')


"""Profiling"""
dimensionality.analyze(df, 'system_battery_max_temperature')
granularity.analyze(df, 'system_battery_max_temperature')
distribution.analyze(df, 'system_battery_max_temperature')
stationarity.analyze(df, "system_battery_max_temperature")

"""Transformation: Analysis"""
scaling.analyze(df, 'system_battery_max_temperature')

# Analyze aggregation with scaled dataset
df_scaled = transform(df, { "scaling": True })
aggregation.analyze(transform(df, { "scaling": True }), 'system_battery_max_temperature')

# Analyze differentiation with scaled and aggregated dataset
df_scaled_hour = transform(df, { "scaling": True, "aggregation": { 'rule': 'h'} })
differentiation.analyze(df_scaled_hour, 'system_battery_max_temperature')

smoothing.analyze(df_scaled_hour, 'system_battery_max_temperature', [12,24,36,48])

"""Transformation: Application"""
df_hour = transform(
    df,
    { "scaling": True, "aggregation": { 'rule': 'h'}, "differentiation": False },
)

"""Modeling"""
options = {"training_pct": 0.80, "smoothing": False}
# options_smoothing = {"training_pct": 0.80, "smoothing": { "window": 12 }}

simple_average.run(df_hour, 'system_battery_max_temperature', options, path='temp/simple-average')
persistence_optimistic.run(df_hour, 'system_battery_max_temperature', options, path='temp/persistence-optimistic')
persistence_realist.run(df_hour, 'system_battery_max_temperature', options, path='temp/persistence-realist')
linear_regression.run(df_hour, 'system_battery_max_temperature', options, path='temp/linear-regression')
rolling_mean.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/rolling-mean-r2')
exponential_smoothing.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/exponential-smoothing-r2-exp')
arima.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/arima')
arima.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2', 'exogenous': ['system_grid_session_duration', 'system_battery_soc'] }, path='temp/arima-exog')
lstm.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/lstm')
lstm_exog.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/lstm-exog')
