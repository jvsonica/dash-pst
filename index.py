import os
import pandas as pd
from preprocess import preprocess
# from profiling import dimensionality, granularity, distribution, stationarity
from transformation import aggregation, differentiation, smoothing
# from pipelines import (
#     simple_average,
#     persistence_optimistic,
#     persistence_realist,
#     linear_regression,
#     # rolling_mean,
#     arima,
#     transform,
# )


def load_data(dir):
    months = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            months.append(df)

    df = pd.concat(months, axis=0, ignore_index=True)
    df = preprocess(df, datetime_col="registered_at")
    # df = aggregation.run(df, gran_level='H')
    return df

df = load_data("./data/1min")

# Using this when we have already loaded and preprocessing the df locally
# to save time on each execution.
# df = pd.read_pickle('./data/resampled-1hr.pkl')

# LA: For the project I will be using "system_battery_max_temperature"
# as the target variable.
# Still, here are other variables in the dataset that may be interesting:
# - system_battery_soc
# - system_battery_min_temperature
# - system_fibo_temperature
# - system_load_controller_igbt_temperature_1
# - system_battery_dchg_kwh

# Profiling
# dimensionality.analyze(df, 'system_battery_max_temperature')
# granularity.analyze(df, 'system_battery_max_temperature')
# distribution.analyze(df, 'system_battery_max_temperature')
# stationarity.analyze(df, "system_battery_max_temperature")
""""
When analyzing the series aggregated by the hour, the series is evaluated as
stationary.
This means that the  statistical properties remain constant over time (there is
no seasonality or trend).

When analyzing the series aggregated by the day, the series is evaluated as
non-stationary.
This means that with a aggregation by the day, temporal dependencies can be 
identified.

- usar agg 1hr com mean -> non-stationary
- usar agg 1hr com max -> non-stationary
- usar agg 1day com max -> stationary
- usar agg 1day com mean -> non-stationary
faz sentido agregar com max? 

transformations:
agg: comparing with Linear Regression 

smoothing:
- with mean? with max?
"""

# Transformation: Analysis
# aggregation.analyze(df, 'system_battery_max_temperature')
# differentiation.analyze(df, 'system_battery_max_temperature')
# smoothing.analyze(df, 'system_battery_max_temperature', [12*60,24*60,36*60,48*60])

# Transformation: Application
# df_hour = transform(
#     df,
#     { "scaling": False, "aggregation": { 'rule': 'h'}, "differentiation": False },
# )

# df_day = transform(
#     df,
#     { "scaling": False, "aggregation": { 'rule': 'd'}, "differentiation": False },
# )

# Modeling
# options = {"training_pct": 0.8, "smoothing": False}
# options_smoothing = {"training_pct": 0.8, "smoothing": { "window": 12 }}

# simple_average.run(df_hour, 'system_battery_max_temperature', options, path='temp/simple-average')
# simple_average.run(df_hour, 'system_battery_max_temperature', options_smoothing, path='temp/simple-average_smoothing')
# persistence_optimistic.run(df_hour, 'system_battery_max_temperature', options, path='temp/persistence-optimistic')
# persistence_optimistic.run(df_hour, 'system_battery_max_temperature', options_smoothing, path='temp/persistence-optimistic-smoothing')
# persistence_realist.run(df_hour, 'system_battery_max_temperature', options, path='temp/persistence-realist')
# persistence_realist.run(df_hour, 'system_battery_max_temperature', options_smoothing, path='temp/persistence-realist-smoothing')
# linear_regression.run(df_hour, 'system_battery_max_temperature', options, path='temp/linear-regression')
# linear_regression.run(df_hour, 'system_battery_max_temperature', options_smoothing, path='temp/linear-regression-smoothing')
# rolling_mean.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/rolling-mean-r2')
# rolling_mean.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'MAPE' }, path='temp/rolling-mean-mape')
# arima.run(df_hour, 'system_battery_max_temperature', { **options }, path='temp/arima')
