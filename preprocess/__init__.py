import pandas as pd
from preprocess.meteo import add_meteo_station

def preprocess(df: pd.DataFrame, datetime_col: str):
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
    df = df.sort_values(datetime_col)
    df = df.set_index(datetime_col)

    # Remove entries in which some important columns are missing
    for col in ['system_battery_max_temperature']:
        df = df.loc[~df[col].isna()]

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
}
