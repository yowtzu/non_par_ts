# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import urllib.request


DATA_DIR = 'data/'

## (Down)load SFO weather data 

for year in range(1987, 2019):
    print(f"Downloading data for year {year}")
    urllib.request.urlretrieve(f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/72494023234.csv", 
                               DATA_DIR + f"{year}-SFO-weather.csv")
    
    
## Download Tides data 

month_ends = pd.date_range(start='1995-12-31', end=pd.datetime.today(), freq='M')
month_starts = month_ends + pd.Timedelta('1d')
month_ends = month_ends[1:]
month_starts = month_starts[:-1]

tides = pd.DataFrame()
for start, end in zip(month_starts, month_ends):
    start_str = f'{start.year}{start.month:02d}{start.day:02d}'
    end_str = f'{end.year}{end.month:02d}{end.day:02d}'
    
    print(f'downloading from {start_str} to {end_str}')
    
    df = pd.read_csv(f'https://tidesandcurrents.noaa.gov/api/datagetter?product=water_level'+
                f'&application=NOS.COOPS.TAC.WL&begin_date={start_str}&end_date={end_str}&'+
                f'datum=MLLW&station=9415020&time_zone=lst&units=metric&format=csv', index_col=0)
    
    tides = pd.concat([tides, df])

tides.index = pd.to_datetime(tides.index)
tides.to_csv(f'point_reyes_tides_{start_str}_{end_str}.csv.gz', compression='gzip')


## Download tides predictions 

start_day = '1995-12-31'
end_day = pd.datetime.today()
month_ends = pd.date_range(start=start_day, end=end_day, freq='M')
month_starts = month_ends + pd.Timedelta('1d')
month_ends = month_ends[1:]
month_starts = month_starts[:-1]

noaa_predictions = pd.DataFrame()
for start, end in zip(month_starts, month_ends):
    start_str = f'{start.year}{start.month:02d}{start.day:02d}'
    end_str = f'{end.year}{end.month:02d}{end.day:02d}'
    
    print(f'downloading from {start_str} to {end_str}')
    
    df = pd.read_csv(f'https://tidesandcurrents.noaa.gov/api/datagetter?product=predictions'+
                f'&application=NOS.COOPS.TAC.WL&begin_date={start_str}&end_date={end_str}&'+
                f'datum=MLLW&station=9415020&time_zone=lst&units=metric&format=csv', index_col=0)
    
    noaa_predictions = pd.concat([noaa_predictions, df])

noaa_predictions.index = pd.to_datetime(noaa_predictions.index)
noaa_predictions.to_csv(f'point_reyes_noaa_predictions_{start_day}_{end_day}.csv.gz', compression='gzip')
