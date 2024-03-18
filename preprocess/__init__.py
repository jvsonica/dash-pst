import pandas as pd
from preprocess.meteo import add_meteo_station

def preprocess(df: pd.DataFrame, datetime_col: str):
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df['system_grid_available'] = df.system_grid_available.astype(float).ffill(limit=120).bfill(limit=120).fillna(0)
    df['system_grid_session_duration'] = (
        df.groupby((df['system_grid_available'] != df['system_grid_available'].shift(1)).cumsum())['system_grid_available'].cumsum()
    )

    df = df.sort_values(datetime_col)
    df = df.set_index(datetime_col)

    # If battery_voltage is 0, then batteries are not properly communicating
    # their metrics, and such records should be dropped.
    df = df.loc[df['system_battery_voltage'] != 0]

    # Remove entries that duplicate the index, probably due to being in 
    # multiple csv's. 
    df = df[~df.index.duplicated(keep='first')]

    # Ensure df has an entry per minute using prefered
    df = df.asfreq('min')

    # Fix target variable missing values. Assume 30 min old values are still good and interpolate
    # for remaining gaps.
    df['system_battery_max_temperature'] = df['system_battery_max_temperature'].bfill(limit=30).ffill(limit=30)
    df['system_battery_max_temperature'] = df.system_battery_max_temperature.interpolate('cubic')

    df['system_battery_soc'] = df['system_battery_soc'].bfill(limit=30).ffill(limit=30)
    df['system_battery_soc'] = df.system_battery_soc.interpolate('linear')

    ## Add meteorology station to dataframe if it doesn't exist yet
    # if 'station_code' not in df.columns:
    #     df = add_meteo_station(df)

    return df


# Aggregation function that should be used in each dataset feature.
aggregation_func_by_col = {
    "system_battery_max_temperature": "max",
    "system_battery_min_temperature": "min",
    "system_battery_soc": "median",
    "system_battery_current": "mean",
    "system_battery_voltage": "mean",
    "system_battery_chg_kwh": "sum",
    "system_battery_dchg_kwh": "sum",
    "system_fibo_temperature": "mean",
    "system_load_controller_igbt_temperature_1": "mean",
    "vehicle_speed_gps": "mean",
    "system_grid_session_duration": "sum"
}
