import pandas as pd
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











