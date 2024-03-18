import os
import pandas as pd
from preprocess import preprocess
from profiling import dimensionality, granularity, distribution, stationarity
from transformation import aggregation, differentiation, smoothing, scaling
from pipelines import (
    simple_average,
    persistence_optimistic,
    persistence_realist,
    linear_regression,
    rolling_mean,
    exponential_smoothing,
    arima,
    lstm
)
from pipelines.tasks import transform


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

# df_b = load_data("./data/72")

# Using this when we have already loaded and preprocessed the df locally
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


"""

# Transformation: Analysis
# scaling.analyze(df, 'system_battery_max_temperature')
# df_scaled = transform(df, { "scaling": True })
# aggregation.analyze(df_scaled, 'system_battery_max_temperature')

# df_hour = transform(df, { "scaling": True, "aggregation": { 'rule': 'h'} })
# differentiation.analyze(df_hour, 'system_battery_max_temperature')
# smoothing.analyze(df, 'system_battery_max_temperature', [12*60,24*60,36*60,48*60])
# smoothing.analyze(df_hour, 'system_battery_max_temperature', [12,24,36,48])

# Transformation: Application
df_hour = transform(
    df,
    { "scaling": True, "aggregation": { 'rule': 'h'}, "differentiation": False },
)

# df_day = transform(
#     df,
#     { "scaling": True, "aggregation": { 'rule': 'd'}, "differentiation": False },
# )

# Modeling
options = {"training_pct": 0.80, "smoothing": False}
options_smoothing = {"training_pct": 0.80, "smoothing": { "window": 24 }}

# simple_average.run(df_hour, 'system_battery_max_temperature', options, path='temp/simple-average')
# persistence_optimistic.run(df_hour, 'system_battery_max_temperature', options, path='temp/persistence-optimistic')
# persistence_realist.run(df_hour, 'system_battery_max_temperature', options, path='temp/persistence-realist')
# linear_regression.run(df_hour, 'system_battery_max_temperature', options, path='temp/linear-regression')
# rolling_mean.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/rolling-mean-r2')
# exponential_smoothing.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/exponential-smoothing-r2-exp')

# simple_average.run(df_hour, 'system_battery_max_temperature', options_smoothing, path='temp/smoothing-simple-average')
# persistence_optimistic.run(df_hour, 'system_battery_max_temperature', options_smoothing, path='temp/smoothing-persistence-optimistic')
# persistence_realist.run(df_hour, 'system_battery_max_temperature', options_smoothing, path='temp/smoothing-persistence-realist')
# linear_regression.run(df_hour, 'system_battery_max_temperature', options_smoothing, path='temp/smoothing-linear-regression')
# rolling_mean.run(df_hour, 'system_battery_max_temperature', { **options_smoothing, 'optimize_for': 'R2' }, path='temp/smoothing-rolling-mean-r2')
# exponential_smoothing.run(df_hour, 'system_battery_max_temperature', { **options_smoothing, 'optimize_for': 'R2' }, path='temp/smoothing-exponential-smoothing-r2-exp')

# arima.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/arima')
arima.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2', 'exogenous': ['system_grid_session_duration', 'system_battery_soc'] }, path='temp/arima-exog')
# (p=2, d=0, q=5)
# lstm.run(df_hour, 'system_battery_max_temperature', { **options, 'optimize_for': 'R2' }, path='temp/lstm')

"""
Final analysis:

<table with R2 results>

During data profiling, significant temperature fluctuations were expected, especially during hotter months.
However, data quality concerns emerged, including missing value periods spanning entire weeks in August-September 2023.
While models themselves are agnostic to the data they're fed, the quality the training set can affect their performance.
It's worth noting that many of these data issues were observed in the training set.

Despite these challenges, basic models like Simple Average and Linear Regression yielded R2 values close to 0.
While these simple models may struggle to adapt to the complex patterns present in the data, they are very useful to benchmark more complex models.

In these more complex models, ARIMA stands out (...)


odel showed promise, its forecasting performance may fall short of the desired accuracy for hourly temperature predictions in storage systems. To improve accuracy, consider implementing very-short term predictions with higher temporal resolution data and incorporating additional exogenous variables, such as ambient temperature forecasts. 

"""