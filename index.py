import pandas as pd
from preprocess import preprocess
from profiling import dimensionality, granularity, distribution, stationarity
from pipelines import simple_average
from pipelines.prepare import prepare


jan = pd.read_csv('./data/1min/236_2023-01.csv')
feb = pd.read_csv('./data/1min/236_2023-02.csv')
mar = pd.read_csv('./data/1min/236_2023-03.csv')
df = pd.concat([jan, feb, mar], axis=0)
df = preprocess(df, datetime_col='registered_at')

# LA: Other variables in the dataset that we could try:
# - system_battery_soc
# - system_battery_max_temperature
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
df_1 = prepare(df, {
    'scaling': False,
    'aggregation': False,
    'differentiation': False,
})

df_2 = prepare(df, {
    'scaling': False,
    'aggregation': { 'rule': 'H' },
    'differentiation': False,
})

# Modeling
simple_average.run(df_1, 'system_battery_max_temperature', { 'training_pct': 0.8, 'smoothing': False }, path='temp/no-transformations')
simple_average.run(df_2, 'system_battery_max_temperature', {'training_pct': 0.8, 'smoothing': False }, path='temp/agg-by-hour')
simple_average.run(df_1, 'system_battery_max_temperature', {'training_pct': 0.8, 'smoothing': { 'window': 30 } }, path='temp/agg-by-hour-w-smoothing')
