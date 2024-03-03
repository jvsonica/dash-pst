import pandas as pd
from preprocess import preprocess
from profiling import dimensionality, granularity, distribution, stationarity
from transformation import aggregation
from pipelines import (
    simple_average,
    persistence_optimistic,
    persistence_realist,
    linear_regression,
    transform
)

months = [
    pd.read_csv('./data/1min/236_2023-01.csv'),
    pd.read_csv('./data/1min/236_2023-02.csv'),
    pd.read_csv('./data/1min/236_2023-03.csv'),
    # pd.read_csv('./data/1min/236_2023-04.csv'),
    # pd.read_csv('./data/1min/236_2023-05.csv'),
    # pd.read_csv('./data/1min/236_2023-06.csv'),
    # pd.read_csv('./data/1min/236_2023-07.csv'),
    # pd.read_csv('./data/1min/236_2023-08.csv'),
    # pd.read_csv('./data/1min/236_2023-09.csv'),
    # pd.read_csv('./data/1min/236_2023-10.csv'),
    # pd.read_csv('./data/1min/236_2023-11.csv'),
    # pd.read_csv('./data/1min/236_2023-12.csv')
]
df = pd.concat(months, axis=0)
df = preprocess(df, datetime_col='registered_at')
df = aggregation.run(df, gran_level='H')

# LA: For the project I will be using "system_battery_max_temperature"
# as the target variable. 
# Still, here are other variables in the dataset that may be interesting:
# - system_battery_soc
# - system_battery_min_temperature
# - system_fibo_temperature
# - system_load_controller_igbt_temperature_1
# - system_battery_dchg_kwh

# Profiling
dimensionality.analyze(df, 'system_battery_max_temperature')
granularity.analyze(df, 'system_battery_max_temperature')
distribution.analyze(df, 'system_battery_max_temperature')
stationarity.analyze(df, 'system_battery_max_temperature')

# Transformation
df_raw = transform(df, {
    'scaling': False,
    'aggregation': False,
    'differentiation': False,
})

# df_scaled = transform(df, {
#     'scaling': True,
#     'aggregation': False,
#     'differentiation': False,
# })

# df_diff = transform(df, {
#     'scaling': False,
#     'aggregation': False,
#     'differentiation': True,
# })

# Modeling
options = { 'training_pct': 0.8, 'smoothing': False }

# simple_average.run(df_scaled, 'system_battery_max_temperature', options, path='temp/scaled-simple-average')
# simple_average.run(df_diff, 'system_battery_max_temperature', options, path='temp/diff-simple-average')
simple_average.run(df_raw, 'system_battery_max_temperature', options, path='temp/raw-simple-average')
persistence_optimistic.run(df_raw, 'system_battery_max_temperature', options, path='temp/raw-persistence-optimistic')
persistence_realist.run(df_raw, 'system_battery_max_temperature', options, path='temp/raw-persistence-realist')
linear_regression.run(df_raw, 'system_battery_max_temperature', options, path='temp/raw-linear-regression')
