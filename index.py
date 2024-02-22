import pandas as pd
from preprocess import preprocess
from profiling import dimensionality, granularity, distribution


jan = pd.read_csv('./data/1min/236_2023-01.csv')
feb = pd.read_csv('./data/1min/236_2023-02.csv')
mar = pd.read_csv('./data/1min/236_2023-03.csv')

df = pd.concat([jan, feb, mar], axis=0)
df = preprocess(df, datetime_col='registered_at')

# LA: Other variables in the dataset that we could try:
# - system_battery_soc
# - system_battery_min_temperature
# - system_dci_temperature
# - system_fibo_temperature
# - system_load_controller_igbt_1_temperature
# - system_load_controller_igbt_2_temperature

# dimensionality.analyze(df, 'system_battery_max_temperature')
# granularity.analyze(df, 'system_battery_max_temperature')
distribution.analyze(df, 'system_battery_max_temperature')