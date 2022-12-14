from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from tabular_data import AirbnbLoader
import numpy as np  
import pandas as pd

def split_data(data):
    X, y = data
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25, random_state=50)
    return X_train, X_test, y_train, y_test



class Model():
  def __init__(self, Model) -> None:
    self.pipeline = Pipeline([('scaler',  StandardScaler()), ('model', Model(random_state = 50))])
  
  def custom_tune_regression_model_hyperparameters(self, X, y):
    loss=['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    penalty = ['l1', 'l2', 'elasticnet']
    alpha = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    shuffle = [True, False]
    learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']

    param_grid = dict(model__loss=loss, model__penalty=penalty, model__alpha=alpha, model__shuffle=shuffle, model__learning_rate=learning_rate)

    grid_model = GridSearchCV(estimator=self.pipeline, param_grid=param_grid, scoring='r2')

    grid_model.fit(X, y)
    
    return grid_model

  def fit_default_model(self, X, y):
    fitted_model = self.pipeline.fit(X,y)
    return fitted_model



def evaluate_performance(name, model, X_test, y_test, X_train, y_train):
  test_predictions = model.predict(X_test)
  train_predictions = model.predict(X_train)
  
  print(f'Performance of model: {name}')
  print('--------------')
  print('TRAINING DATA:')
  print(f'R2 score: {r2_score(y_train, train_predictions)}')
  print(f'MSE: {mean_squared_error(y_train, train_predictions)}')
  print('TEST DATA:')
  print(f'R2 score: {r2_score(y_test, test_predictions)}')
  print(f'MSE: {mean_squared_error(y_test, test_predictions)}')
  print(' ')



if __name__ == "__main__":
  # Load data
  loader = AirbnbLoader()
  init_data = loader.load_airbnb('Price_Night', normalized=True)
  X_train, X_test, y_train, y_test = split_data(init_data)

  # Create models
  model = Model(SGDRegressor)
  tuned_model = model.custom_tune_regression_model_hyperparameters(X=X_train, y=y_train)
  default_model = model.fit_default_model(X=X_train, y=y_train)

  # Evaluate performance
  evaluate_performance('Tuned model', tuned_model, X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)
  evaluate_performance('Default model', default_model, X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train)