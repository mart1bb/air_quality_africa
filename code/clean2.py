import pandas as pd
import numpy as np
import time


def restructure(path):
    data = pd.read_csv(path, sep=';',chunksize=30000)
    dfs = []

    for chunk in data:
        df = pd.DataFrame()
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'],format='ISO8601').round('min')
        
        chunk = chunk.loc[(chunk['value_type'] == 'P0') | 
                        (chunk['value_type'] == 'P1') | 
                        (chunk['value_type'] == 'P2') | 
                        (chunk['value_type'] == 'temperature') | 
                        (chunk['value_type'] == 'humidity')]
        
        chunk['value'] = pd.to_numeric(chunk['value'],errors='coerce')

        df = pd.pivot_table(chunk, values=['value'], index=['timestamp'],columns=['value_type'], aggfunc=np.mean)
        df = df.droplevel(0,axis=1).rename_axis(columns=None)

        chunk.drop_duplicates(subset='timestamp',inplace=True)
        chunk.set_index('timestamp',inplace=True)

        df['lat'] = chunk[['lat']]
        df['lon'] = chunk[['lon']]

        dfs.append(df.reset_index())

    final_df = pd.concat(dfs)

    return final_df

def restructure_parquet(path):
    data = pd.read_parquet(path)

    data['timestamp'] = pd.to_datetime(data['timestamp'],format='ISO8601').round('min')

    data = data.loc[(data['value_type'] == 'P0') | 
                    (data['value_type'] == 'P1') | 
                    (data['value_type'] == 'P2') | 
                    (data['value_type'] == 'temperature') | 
                    (data['value_type'] == 'humidity')]

    df = pd.pivot_table(data, values=['value'],index=['timestamp','location','lat','lon'],columns=['value_type'], aggfunc=np.sum)
    df = df.droplevel(0,axis=1).rename_axis(columns=None)
    df.reset_index(inplace=True)

    return df

# start = time.time()
# print(restructure_parquet('./raw/nairobi_parquet/december_2019_sensor_data_archive.parquet'))
# end = time.time()
# elapsed = end - start
# print(f'Temps d\'exécution : {elapsed}s')














