import pandas as pd
import time
import os
import clean2 as clean
from progress.bar import Bar

start = time.time()

dfs = []

city = "nairobi_parquet"
names = os.listdir('E:/Code/VisualStudioPy/MachineLearning/Projet/raw/' + city)

print("Cleaning data")
bar = Bar('Classifying...',max=len(names))
for name in names:
    try:
        dfs.append(clean.restructure_parquet('./raw/' + city + '/' + name))
    except:
        print(" Error to Classify =>",name)
    bar.next()

final_df = pd.concat(dfs) # .drop_duplicates(subset='timestamp').set_index('timestamp')

print(final_df)

final_df.to_parquet('./cleaned/restructured3.parquet')

end = time.time()
elapsed = end - start
print(f'Temps d\'exécution : {elapsed}s')