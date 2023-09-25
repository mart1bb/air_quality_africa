import pandas as pd
<<<<<<< HEAD
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
import matplotlib.pyplot as plt

# models : in => time, humidity, temperature

# model 1 : out => P0
# model 2 : out => P1
# model 3 : out => P2

df = pd.read_parquet("./cleaned/restructured3.parquet")

df1 = df[['timestamp','lat','lon','humidity','temperature','P0']].dropna()
df2 = df[['timestamp','lat','lon','humidity','temperature','P1']].dropna()
df3 = df[['timestamp','lat','lon','humidity','temperature','P2']].dropna()

y1 = df1['P0']
y2 = df2['P1']
y3 = df3['P2']

for dataframe in [df1,df2,df3]:
    dataframe['month'] = dataframe['timestamp'].dt.month
    dataframe['day'] = dataframe['timestamp'].dt.day
    dataframe['hour'] = dataframe['timestamp'].dt.hour
    dataframe['minute'] = dataframe['timestamp'].dt.minute
    dataframe.drop(['timestamp'],axis=1,inplace=True)
    # dataframe.pivot_table(values=['P0'],index=['lat','lon','humidity','temperature',''],columns=['month','day','hour','minute'])

X1 = df1[['month','day','hour','minute','lat','lon','humidity','temperature']]
X2 = df2[['month','day','hour','minute','lat','lon','humidity','temperature']]
X3 = df3[['month','day','hour','minute','lat','lon','humidity','temperature']]

x_train1, x_test1, y_train1, y_test1 = train_test_split(X1,y1, train_size=0.8,shuffle=False)
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2,y2, train_size=0.8,shuffle=False)
x_train3, x_test3, y_train3, y_test3 = train_test_split(X3,y3, train_size=0.8,shuffle=False)


# decision_tree_regressor_P0 = DecisionTreeRegressor()
# decision_tree_regressor_P0.fit(x_train1,y_train1)

# print(len(y_train1))
# print(decision_tree_regressor_P0.score(x_test1,y_test1))


# # Définissons l'architecture du modèle Ann
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim=len(X1.columns), activation='relu')) # Couche cachée avec 16 neurones et fonction d'activation ReLU
model.add(tf.keras.layers.Dense(1, activation='linear')) # Couche de sortie avec 1 neurone et fonction d'activation linéaire

#Entrainons le modèle sur nos données train
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train1, y_train1, epochs=10, batch_size=100)

loss = model.evaluate(x_test1, y_test1)
print("MSE sur les données de test : ", loss)

loss_values = history.history['loss']
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()








=======
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter 

# models : in => time, humidity, temperature

# model 1 : out => P0
# model 2 : out => P1
# model 3 : out => P2

epochs = 30
batch_size = 100
>>>>>>> building_coherent_data

df = pd.read_parquet("./coherent/location_cut_incoherent_temp.parquet")

df1 = df[['timestamp','lat','lon','humidity','temperature','P0']].dropna()
df2 = df[['timestamp','lat','lon','humidity','temperature','P1']].dropna()
df3 = df[['timestamp','lat','lon','humidity','temperature','P2']].dropna()

<<<<<<< HEAD
=======
y1 = savgol_filter(df1['P0'],20,3)
y2 = savgol_filter(df2['P1'],20,3)
y3 = savgol_filter(df3['P2'],20,3)

for dataframe in [df1,df2,df3]:
    dataframe['month'] = dataframe['timestamp'].dt.month
    dataframe['day'] = dataframe['timestamp'].dt.day
    dataframe['hour'] = dataframe['timestamp'].dt.hour
    dataframe['minute'] = dataframe['timestamp'].dt.minute
    dataframe.drop(['timestamp'],axis=1,inplace=True)
    # dataframe.pivot_table(values=['P0'],index=['lat','lon','humidity','temperature',''],columns=['month','day','hour','minute'])

X1 = df1[['month','day','hour','minute','lat','lon','humidity','temperature']]
X2 = df2[['month','day','hour','minute','lat','lon','humidity','temperature']]
X3 = df3[['month','day','hour','minute','lat','lon','humidity','temperature']]

x_train1, x_test1, y_train1, y_test1 = train_test_split(X1,y1, train_size=0.8,shuffle=True)
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2,y2, train_size=0.8,shuffle=False)
x_train3, x_test3, y_train3, y_test3 = train_test_split(X3,y3, train_size=0.8,shuffle=False)

# Définissons l'architecture du modèle Ann
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Dense(8, input_dim=len(X1.columns), activation='relu'))
model1.add(tf.keras.layers.Dense(12, input_dim=len(X1.columns), activation='relu'))
model1.add(tf.keras.layers.Dense(16, input_dim=len(X1.columns), activation='relu'))
model1.add(tf.keras.layers.Dense(8, input_dim=len(X1.columns), activation='relu'))
model1.add(tf.keras.layers.Dense(4, input_dim=len(X1.columns), activation='relu'))
model1.add(tf.keras.layers.Dense(1, activation='linear')) # Couche de sortie avec 1 neurone et fonction d'activation linéaire

#Entrainons le modèle sur nos données train
model1.compile(loss='mean_squared_error', optimizer='adam')
history1 = model1.fit(x_train1, y_train1, epochs=epochs, batch_size=batch_size)

loss1 = model1.evaluate(x_test1, y_test1)
print("MSE sur les données de test : ", loss1)

# Définissons l'architecture du modèle Ann
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(8, input_dim=len(X1.columns), activation='relu'))
model2.add(tf.keras.layers.Dense(12, input_dim=len(X1.columns), activation='relu'))
model2.add(tf.keras.layers.Dense(16, input_dim=len(X2.columns), activation='relu')) # Couche cachée avec 16 neurones et fonction d'activation ReLU
model2.add(tf.keras.layers.Dense(8, input_dim=len(X1.columns), activation='relu'))
model2.add(tf.keras.layers.Dense(4, input_dim=len(X1.columns), activation='relu'))
model2.add(tf.keras.layers.Dense(1, activation='linear')) # Couche de sortie avec 1 neurone et fonction d'activation linéaire

#Entrainons le modèle sur nos données train
model2.compile(loss='mean_squared_error', optimizer='adam')
history2 = model2.fit(x_train2, y_train2, epochs=epochs, batch_size=batch_size)

loss2 = model2.evaluate(x_test2, y_test2)
print("MSE sur les données de test : ", loss2)

# Définissons l'architecture du modèle Ann
model3 = tf.keras.Sequential()
model3.add(tf.keras.layers.Dense(8, input_dim=len(X1.columns), activation='relu'))
model3.add(tf.keras.layers.Dense(12, input_dim=len(X1.columns), activation='relu'))
model3.add(tf.keras.layers.Dense(16, input_dim=len(X3.columns), activation='relu')) # Couche cachée avec 16 neurones et fonction d'activation ReLU
model3.add(tf.keras.layers.Dense(8, input_dim=len(X1.columns), activation='relu'))
model3.add(tf.keras.layers.Dense(4, input_dim=len(X1.columns), activation='relu'))
model3.add(tf.keras.layers.Dense(1, activation='linear')) # Couche de sortie avec 1 neurone et fonction d'activation linéaire

#Entrainons le modèle sur nos données train
model3.compile(loss='mean_squared_error', optimizer='adam')
history3 = model3.fit(x_train3, y_train3, epochs=epochs, batch_size=batch_size)

loss3 = model3.evaluate(x_test3, y_test3)
print("MSE sur les données de test : ", loss3)

loss_values1 = history1.history['loss']
epochs1 = range(1, len(loss_values1)+1)

loss_values2 = history2.history['loss']
epochs2 = range(1, len(loss_values2)+1)

loss_values3 = history3.history['loss']
epochs3 = range(1, len(loss_values3)+1)

plt.plot(epochs1, loss_values1, label='Training Loss P0')
plt.plot(epochs2, loss_values2, label='Training Loss P1')
plt.plot(epochs3, loss_values3, label='Training Loss P2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
>>>>>>> building_coherent_data
