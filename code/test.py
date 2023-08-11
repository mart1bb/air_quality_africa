import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_parquet('E:/Code/VisualStudioPy/MachineLearning/Projet/raw/nairobi_parquet/august_2018_sensor_data_archive.parquet')

df = pd.DataFrame()



data['timestamp'] = pd.to_datetime(data['timestamp'],format='ISO8601',errors='raise').round('min')

data = data.loc[(data['value_type'] == 'P0') | 
                (data['value_type'] == 'P1') | 
                (data['value_type'] == 'P2') | 
                (data['value_type'] == 'temperature') | 
                (data['value_type'] == 'humidity')]

df = pd.pivot_table(data, values=['value'],index=['timestamp','location','lat','lon'],columns=['value_type'], aggfunc=np.sum)
df = df.droplevel(0,axis=1).rename_axis(columns=None)
df.reset_index(inplace=True)

print(df)

# for location in list(set(df['location'])):
#     temperature = df.loc[df['location'] == location,['timestamp','temperature']]
#     plt.scatter(temperature.timestamp,temperature.temperature,s=0.2,label=location)

# plt.legend()
# plt.gcf().autofmt_xdate()
# plt.show()