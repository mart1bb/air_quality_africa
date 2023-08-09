from sklearn.preprocessing import StandardScaler
import pandas
import pickle

################################Target : temperature ############################
#lecture du fichier json et séparation des données, on procède par chunk car il y a énormement de lignes
chunkSize = 100000
chunkNb = 0
rowX = pandas.DataFrame(columns = ['annee','mois','jour','heure','lon','lat','humidity','P0','P1','P2'])
y = pandas.DataFrame(columns = ["temperature"])

for chunk in pandas.read_json('./Training/trainingDataset.json',lines=True, chunksize=chunkSize):
    newRowX = chunk.drop(['temperature'], axis=1)
    rowX = pandas.concat([rowX,newRowX])
    newY = pandas.DataFrame({"temperature":chunk['temperature']}) #target
    y = pandas.concat([y,newY])
    print(chunkNb)
    if chunkNb == 10:
        break
    chunkNb+=1

print("Read ended...")
ss = StandardScaler().fit(rowX)
normalised = ss.transform(rowX)
x = normalised

ssFile = open('./TrainedModele/temperatureSS.pkl','wb')
pickle.dump(ss, ssFile)
ssFile.close()

#definition des valeurs test et valeurs d'apprentissage
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

from sklearn.model_selection import GridSearchCV
# def regressorError(model,x,y,paramDict,name): # best param is n = 5 with cv = 5
#     grid = GridSearchCV(model(),paramDict,cv=3)
#     grid.fit(x_train,y_train)
#     print(name,' best param : ',grid.best_estimator_,', best score : ',grid.best_score_)


#KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)
y_prediction=knn.predict(x_test)

mae=mean_absolute_error(y_test, y_prediction)
print('score mae=',mae)
mse=mean_squared_error(y_test, y_prediction)
print('score ms=',mse)
r2=r2_score(y_test, y_prediction)
print('score r2=',r2)
# print(knn.score(x_test, y_test))

file = open('./TrainedModele/temperatureModele.pkl', 'wb') 
pickle.dump(knn, file)  
file.close()

#Fin KNN

# #DecisionTree et validation croisée

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import cross_val_score
# for i in range(1,10):
#     regressor = DecisionTreeRegressor(max_depth=i)
#     scores=cross_val_score(regressor, x, y, cv=5)
#     print('Cross-validation scores=',scores)


# from sklearn.model_selection import cross_val_predict
# regressor = DecisionTreeRegressor()# on définit le modele de regression 

# y_prediction2 = cross_val_predict(regressor, x, y, cv=5)#prédiction avec validation croisée
# print(y_prediction)


# mae=mean_absolute_error(y, y_prediction2)
# print('score mae=',mae)
# mse=mean_squared_error(y, y_prediction2)
# print('score ms=',mse)
# r2=r2_score(y, y_prediction2)
# print('score r2=',r2)
# #Fin DecisionTree


# #Ann
# import tensorflow as tf
# # Définissons l'architecture du modèle Ann
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(16, input_dim=10, activation='relu')) # Couche cachée avec 16 neurones et fonction d'activation ReLU
# model.add(tf.keras.layers.Dense(1, activation='linear')) # Couche de sortie avec 1 neurone et fonction d'activation linéaire

# #Entrainons le modèle sur nos données train
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=3, batch_size=10)

# loss = model.evaluate(x_test, y_test)
# print("MSE sur les données de test : ", loss)
# #Fin Ann




################################Target : PO #######################

chunkSize = 100000
chunkNb = 0
rowX = pandas.DataFrame(columns = ['annee','mois','jour','heure','lon','lat','humidity','P1','P2','temperature'])
ypo = pandas.DataFrame(columns = ["P0"])


for chunk in pandas.read_json('./Training/trainingDataset.json',lines=True, chunksize=chunkSize):
    newRowX = chunk.drop(['P0'], axis=1)
    rowX = pandas.concat([rowX,newRowX])
    newY = pandas.DataFrame({"P0":chunk['P0']}) #target
    ypo = pandas.concat([ypo,newY])
    print(chunkNb)
    if chunkNb == 10:
        break
    chunkNb+=1

print("Read ended...")

ss = StandardScaler().fit(rowX)
normalised = ss.transform(rowX)
xpo = normalised

ssFile = open('./TrainedModele/P0SS.pkl','wb')
pickle.dump(ss, ssFile)
ssFile.close()

#definition des valeurs test et valeurs d'apprentissage
from sklearn.model_selection import train_test_split
x_train3, x_test3, y_train3, y_test3 = train_test_split(xpo, ypo, train_size=0.8, random_state=1)

#KNN

knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train3, y_train3)
y_prediction3=knn.predict(x_test3)

mae=mean_absolute_error(y_test3, y_prediction3)
print('score mae=',mae)
mse=mean_squared_error(y_test3, y_prediction3)
print('score ms=',mse)
r2=r2_score(y_test3, y_prediction3)
print('score r2=',r2)
print(knn.score(x_test3, y_test3))

file = open('./TrainedModele/P0Modele.pkl', 'wb') 
pickle.dump(knn, file)  
file.close()

#Fin KNN


# #DecisionTree et validation croisée

# for i in range(1,10):
#     regressor = DecisionTreeRegressor(max_depth=i)
#     scores=cross_val_score(regressor, xpo, ypo, cv=5)
#     print('Cross-validation scores=',scores)


# from sklearn.model_selection import cross_val_predict
# regressor = DecisionTreeRegressor()# on définit le modele de regression 

# y_prediction4 = cross_val_predict(regressor, xpo, ypo, cv=5)#prédiction avec validation croisée


# mae=mean_absolute_error(ypo, y_prediction4)
# print('score mae=',mae)
# mse=mean_squared_error(ypo, y_prediction4)
# print('score ms=',mse)
# r2=r2_score(ypo, y_prediction4)
# print('score r2=',r2)
# #Fin DecisionTree


# #Ann

# #Définissons l'architecture du modèle Ann
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(16, input_dim=10, activation='relu')) # Couche cachée avec 16 neurones et fonction d'activation ReLU
# model.add(tf.keras.layers.Dense(1, activation='linear')) # Couche de sortie avec 1 neurone et fonction d'activation linéaire

# #Entrainons le modèle sur nos données train
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train3, y_train3, epochs=3, batch_size=10)

# loss = model.evaluate(x_test3, y_test3)
# print("MSE sur les données de test : ", loss)
# #Fin Ann



#########################Target : humidity ##################################

chunkSize = 100000
chunkNb = 0
rowX = pandas.DataFrame(columns = ['annee','mois','jour','heure','lon','lat','P0','P1','P2','temperature'])
yh = pandas.DataFrame(columns = ["humidity"])

for chunk in pandas.read_json('./Training/trainingDataset.json',lines=True, chunksize=chunkSize):
    newRowX = chunk.drop(['humidity'], axis=1)
    rowX = pandas.concat([rowX,newRowX])
    newY = pandas.DataFrame({"humidity":chunk['humidity']}) #target
    yh = pandas.concat([yh,newY])
    print(chunkNb)
    if chunkNb == 10:
        break
    chunkNb+=1

print("Read ended...")

ss = StandardScaler().fit(rowX)
normalised = ss.transform(rowX)
xh = normalised

ssFile = open('./TrainedModele/humiditySS.pkl','wb')
pickle.dump(ss, ssFile)
ssFile.close()

#definition des valeurs test et valeurs d'apprentissage
from sklearn.model_selection import train_test_split
x_train4, x_test4, y_train4, y_test4 = train_test_split(xh, yh, train_size=0.8, random_state=1)

#KNN

knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train4, y_train4)
y_prediction5=knn.predict(x_test4)

mae=mean_absolute_error(y_test4, y_prediction5)
print('score mae=',mae)
mse=mean_squared_error(y_test4, y_prediction5)
print('score ms=',mse)
r2=r2_score(y_test4, y_prediction5)
print('score r2=',r2)
print(knn.score(x_test4, y_test4))

file = open('./TrainedModele/humidityModele.pkl', 'wb') 
pickle.dump(knn, file)  
file.close()

#Fin KNN


# #DecisionTree et validation croisée

# for i in range(1,10):
#     regressor = DecisionTreeRegressor(max_depth=i)
#     scores=cross_val_score(regressor, xh, yh, cv=5)
#     print('Cross-validation scores=',scores)


# from sklearn.model_selection import cross_val_predict
# regressor = DecisionTreeRegressor()# on définit le modele de regression 

# y_prediction4 = cross_val_predict(regressor, xh, yh, cv=5)#prédiction avec validation croisée


# mae=mean_absolute_error(yh, y_prediction4)
# print('score mae=',mae)
# mse=mean_squared_error(yh, y_prediction4)
# print('score ms=',mse)
# r2=r2_score(yh, y_prediction4)
# print('score r2=',r2)
# #Fin DecisionTree


# #Ann

# #Définissons l'architecture du modèle Ann
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(16, input_dim=10, activation='relu')) # Couche cachée avec 16 neurones et fonction d'activation ReLU
# model.add(tf.keras.layers.Dense(1, activation='linear')) # Couche de sortie avec 1 neurone et fonction d'activation linéaire

# #Entrainons le modèle sur nos données train
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train4, y_train4, epochs=3, batch_size=10)

# loss = model.evaluate(x_test4, y_test4)
# print("MSE sur les données de test : ", loss)
# #Fin Ann