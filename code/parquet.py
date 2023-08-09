import pandas as pd
import os


city = "nairobi"
names = os.listdir('D:/marti/Bureau/Algo/Code/VisualStudioPy/MachineLearning/Projet/raw/nairobi')

for name in names:
    df = pd.read_csv('./raw/' + city + '/' + name, sep=';', chunksize=30000)
    temp = []
    for chunk in df:
        chunk['value'] = pd.to_numeric(chunk['value'],errors='coerce')

        temp.append(chunk)
    
    parquet = pd.concat(temp)

    parquet.to_parquet('./raw/' + city + '_parquet/' + name.replace('csv','parquet'))

