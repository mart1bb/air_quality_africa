import time
import pandas
from progress.bar import Bar
import os
import json


def addGen(source,storage,locationData):
    print("")

    bar = Bar("   Ajout de " + source + "...",max=int(sum(1 for row in open(source, 'r'))/30000)+1)

    chunkSize = 30000
    rowIndex = 0
    error = 0
    raw = pandas.read_csv(source,sep=';', chunksize=chunkSize)

    for chunk in raw:
        for i in range(rowIndex,rowIndex+len(chunk.sensor_id)):
            try:
                storage[str(chunk.location[i])][chunk.timestamp[i][:16]][chunk.value_type[i]] = float(chunk.value[i])
            except:
                try:
                    storage[str(chunk.location[i])][chunk.timestamp[i][:16]] = {}
                    storage[str(chunk.location[i])][chunk.timestamp[i][:16]][chunk.value_type[i]] = float(chunk.value[i])
                except:
                    try:
                        locationData[str(chunk.location[i])] = {}
                        locationData[str(chunk.location[i])]["lat"] = chunk.lat[i]
                        locationData[str(chunk.location[i])]["lon"] = chunk.lon[i]
                        storage[str(chunk.location[i])] = {}
                        storage[str(chunk.location[i])][chunk.timestamp[i][:16]] = {}
                        storage[str(chunk.location[i])][chunk.timestamp[i][:16]][chunk.value_type[i]] = float(chunk.value[i])
                    except:
                        error+=1
        bar.next()
        rowIndex += len(chunk.sensor_id)
            
    print("")
    print("   Total Error : ",error)
    return storage,locationData


start = time.time()


city = "nairobi"
names = os.listdir('D:/marti/Bureau/Algo/Code/VisualStudioPy/MachineLearning/Projet/raw/nairobi')


locationFile = open("./test/location.json")
locationData = json.load(locationFile)

file = open("./test/cleaned.json")
storage = json.load(file)

print("Cleaning data")
bar = Bar('Classifying...',max=len(names))
for name in names:
    try:
        storage,locationData = addGen('./raw/' + city + '/' + name,storage,locationData)
        # locationData = classifieData.locationCorrect('./row/' + city + '/' + name + '.csv',locationData)
    except:
        print(" Error to Classify")
    bar.next()
print("")
print("Data cleaned")



end = time.time()
elapsed = end - start
print(f'Temps d\'exécution : {elapsed}s')












