import pandas
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

def regressorError(model,X,y,paramDict,name): # best param is n = 5 with cv = 5
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    grid = GridSearchCV(model(),paramDict,cv=3)
    grid.fit(x_train,y_train)
    print(name,' best param : ',grid.best_estimator_,', best score : ',grid.best_score_)

chunkSize = 100000

for chunk in pandas.read_json('./Training/trainingDataset.json',lines=True, chunksize=chunkSize):
    rowX = chunk.drop(['temperature'], axis=1)
    mpgN = StandardScaler().fit_transform(rowX)
    mpgX = mpgN
    mpgy = chunk['temperature']



    
    break
