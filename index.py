import pandas as pd
from preprocess import preprocess
from profiling import dimensionality, granularity, distribution, stationarity
from transformation import scaling, smoothing, aggregation, differentiation


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

## Profiling
# dimensionality.analyze(df, 'system_battery_max_temperature')
# granularity.analyze(df, 'system_battery_max_temperature')
# distribution.analyze(df, 'system_battery_max_temperature')
# stationarity.analyze(df, 'system_battery_max_temperature')


## Transformations
agg_funcs = {
    'system_battery_max_temperature': 'mean',
    # 'system_battery_min_temperature': 'mean',
    'system_battery_soc': 'median',
    'system_battery_current': 'mean',
    # 'system_battery_voltage': 'mean',
    'system_battery_chg_kwh': 'sum',
    # 'system_battery_dchg_kwh': 'sum',
    # 'system_fibo_temperature': 'mean',
    # 'system_load_controller_igbt_temperature_1': 'mean',
    'vehicle_speed_gps': 'mean',
}
cols = agg_funcs.keys()

# scaled = scaling.run(df[cols], 'system_battery_max_temperature')
# smoothed = smoothing.run(df, 'system_battery_max_temperature', window=50)
# aggregated = aggregation.run(df[cols], 'system_battery_max_temperature', gran_level='W', agg_funcs=agg_funcs)
differentiation.run(df[cols], 'system_battery_max_temperature')