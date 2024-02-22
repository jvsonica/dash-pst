import pandas as pd
from meteostat import Stations, Hourly # https://dev.meteostat.net


def get_closest_station(lat, lon):
    """Find the meteo station closest to `lat`, `lon` coordinates.

    Args:
        lat (float): latitude
        lon (float): longitude

    Returns:
        dict: station information, including "icao" and "name".
    """
    try:
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)
        return station.iloc[0].to_dict()
    except Exception as e:
        print('could not find station', e)
        return None


def add_meteo_station(df):
    """Enriches df with meteostat data with the following new cols:
    - station_code
    - station_name 

    Args:
        df (pd.DataFrame): original dataframe.
    """
    def aggregate(chunk):
        station = get_closest_station(chunk.latitude.mean(), chunk.longitude.mean())
        print('add_meteo_station', chunk.index[0], '/', len(df))
        return pd.DataFrame({
            'station_code': [station['icao']] * len(chunk),
            'station_name': [station['name']] * len(chunk),
        }, index=chunk.index)
    
    print('add_meteo_station', 'adding meteostat information to df')

    grouped = df.groupby(df.index // 60, group_keys=False) # calling aggregate once per hour
    meteo = grouped.apply(aggregate)

    df['station_code'] = meteo['station_code']
    df['station_name'] = meteo['station_name']
    return df
