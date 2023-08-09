import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_parquet('E:/Code/VisualStudioPy/MachineLearning/Projet/raw/nairobi_parquet/april_2018_sensor_data_archive.parquet')
data = data.loc[data['location'] == 7]

df = pd.DataFrame()
data['timestamp'] = pd.to_datetime(data['timestamp'],format='ISO8601').round('min')

data = data.loc[(data['value_type'] == 'P0') | 
                (data['value_type'] == 'P1') | 
                (data['value_type'] == 'P2') | 
                (data['value_type'] == 'temperature') | 
                (data['value_type'] == 'humidity')]

df = pd.pivot_table(data, values=['value'],index=['timestamp'],columns=['value_type','location'], aggfunc=np.sum)
df = df.droplevel(0,axis=1).rename_axis(columns=None)

print(len(df.index)-len(df.index.drop_duplicates()))

# df = data.set_index('timestamp')
print(df)

temperature = df[['temperature']]
plt.scatter(temperature.index,temperature.temperature,s=0.2)
plt.gcf().autofmt_xdate()
plt.show()