import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.signal import savgol_filter 


df = pd.read_parquet("./cleaned/restructured3.parquet")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('location',inplace=True)
df['location'] = df['location'].astype(str)

locations = list(set(df.location))

mean_temperature_nairobi = 18.8
location_indicator = pd.DataFrame(columns=['quantity','quality','mean_temperature'],index=locations)

for loc in locations:
    values = df.loc[df['location'] == loc][['temperature']]
    location_indicator.loc[loc,'mean_temperature'] = np.mean(values)
    location_indicator.loc[loc,'quality'] = (abs(np.mean(values)-mean_temperature_nairobi)/mean_temperature_nairobi)*100
    location_indicator.loc[loc,'quantity'] = (len(values)/len(df.temperature))*100

good_locations = location_indicator.loc[location_indicator['quality'] < 25].index

df_2 = df.loc[df['location'].isin(good_locations)]


print((len(df_2)/len(df))*100)










