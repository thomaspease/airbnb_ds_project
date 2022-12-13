from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tabular_data import AirbnbLoader
import numpy as np  
import pandas as pd

def split_data(data):
    X, y = data
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test


def linear_regression():
  lin_reg = make_pipeline(StandardScaler(), SGDRegressor())
  X, y = load_airbnb('Price_Night')
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
  lin_reg.fit(X_train, y_train)
  train_pred = lin_reg.predict(X_train)
  test_pred = lin_reg.predict(X_test)
  train_mse = mean_squared_error(y_train, train_pred)
  test_mse = mean_squared_error(y_test, test_pred)
  print(train_mse)
  print(test_mse)

def custom_tune_regression_model_hyperparameters(Model, data):
  loss=['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
  penalty = ['l1', 'l2', 'elasticnet']
  alpha = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
  shuffle = [True, False]
  learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']

  param_grid = dict(loss=loss, penalty=penalty, alpha=alpha, shuffle=shuffle, learning_rate=learning_rate)

  init_model = Model()

  model = GridSearchCV(estimator=init_model, param_grid=param_grid, scoring='neg_mean_squared_error')

  model.fit(*data)
  
  # print(model.cv_results_)

def evaluate_performance():
  pass

loader = AirbnbLoader()
init_data = loader.load_airbnb('Price_Night')
X_train, X_test, y_train, y_test = split_data(init_data)

custom_tune_regression_model_hyperparameters(Model=SGDRegressor, data=[X_train, y_train])