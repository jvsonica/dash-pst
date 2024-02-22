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
