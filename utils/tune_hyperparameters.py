from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit, GridSearchCV, RandomizedSearchCV
import os
from joblib import dump, load
import json
from datetime import datetime 

class TuneModel():
  def __init__(self, X, y, model, param_grid, method='grid', n_iter=20):
    if method not in ['grid', 'random']:
      raise ValueError('Invalid method, must be either "random" or "grid"')

    self.data = {}
    self.data['X'] = X
    self.data['y'] = y
    self.model_class = model
    self.param_grid = param_grid
    self.method = method

    self.cv_split = ShuffleSplit(n_splits = 5, test_size = .3, train_size = .7, random_state = 42)

  def return_tuned_results(self, model):
    data = {
    'params' : model.best_params_,
    'train' : model.cv_results_["mean_train_score"][model.best_index_],
    'test': model.cv_results_["mean_test_score"][model.best_index_]
    }
    return data


# This is just working with accuracy at the moment as was built for multiclass classification
class ClassificationTuneModel(TuneModel):
  def __init__(self, X, y, model, param_grid, method='grid', n_iter=20):
    super().__init__(X, y, model, param_grid, method, n_iter)

    self.default_model_performance_ = self.__return_default_results()

    self.tuned_model = self.__tune_and_return_model(n_iter)

    self.tuned_performance_ = self.return_tuned_results(self.tuned_model)

  def __return_default_results(self):
    default_model = self.model_class()

    base_results = cross_validate(default_model, self.data['X'], self.data['y'], cv = self.cv_split, scoring='accuracy', return_train_score=True)

    default_model.fit(self.data['X'], self.data['y']) #Just for defaults params

    data = {
      'params' : default_model.get_params(),
      'train accuracy' : base_results["train_score"].mean(),
      'test accuracy': base_results["test_score"].mean(),
    }

    return data
  
  def __tune_and_return_model(self, n_iter):    
    if self.method == 'grid':
      tuned_model = GridSearchCV(self.model_class(), param_grid=self.param_grid, scoring='accuracy', cv=self.cv_split, return_train_score=True)
    if self.method == 'random':
      tuned_model = RandomizedSearchCV(self.model_class(), param_distributions=self.param_grid, n_iter=n_iter, scoring='accuracy', cv=self.cv_split, return_train_score=True)

    tuned_model.fit(self.data['X'], self.data['y'])

    return tuned_model
  
  def save_model_and_results(self):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M")

    path = f'/Users/tompease/Documents/Coding/airbnb/models/classification/{self.model_class.__name__}/{dt_string}'

    if not os.path.exists(path):
      os.makedirs(path)

    dump(self.tuned_model, f'{path}/model.joblib')

    with open(f'{path}/metrics.json', "w") as outfile:
      json.dump(self.tuned_performance_, outfile, indent=4)
    
    

class RegressionTuneModel(TuneModel):
  def __init__(self, X, y, model, param_grid, method='grid', n_iter=20):
    super().__init__(X, y, model, param_grid, method, n_iter)
    
    self.default_model_performance_ = self.__return_default_results()

    self.mse_tuned_model = self.__tune_and_return_model('neg_mean_squared_error', n_iter)
    self.r2_tuned_model = self.__tune_and_return_model('r2', n_iter)

    self.tuned_for_mse_performance_ = self.return_tuned_results(self.mse_tuned_model)
    self.tuned_for_r2_performance_ = self.return_tuned_results(self.r2_tuned_model)

  def save_model_and_results(self):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M")

    mse_path = f'/Users/tompease/Documents/Coding/airbnb/models/regression/{self.model_class.__name__}/{dt_string}/tuned_for_mse'
    r2_path = f'/Users/tompease/Documents/Coding/airbnb/models/regression/{self.model_class.__name__}/{dt_string}/tuned_for_r2'
    
    for path in [mse_path, r2_path]:
      if not os.path.exists(path):
        os.makedirs(path)

    dump(self.mse_tuned_model, f'{mse_path}/model.joblib')
    dump(self.r2_tuned_model, f'{r2_path}/model.joblib')

    with open(f'{mse_path}/metrics.json', "w") as outfile:
      json.dump(self.tuned_for_mse_performance_, outfile, indent=4)
    
    with open(f'{r2_path}/metrics.json', "w") as outfile:
      json.dump(self.tuned_for_r2_performance_, outfile, indent=4)
    

  def __tune_and_return_model(self, scoring, n_iter):    
    if self.method == 'grid':
      tuned_model = GridSearchCV(self.model_class(), param_grid=self.param_grid, scoring=scoring, cv=self.cv_split, return_train_score=True)
    if self.method == 'random':
      tuned_model = RandomizedSearchCV(self.model_class(), param_distributions=self.param_grid, n_iter=n_iter, scoring=scoring, cv=self.cv_split, return_train_score=True)

    tuned_model.fit(self.data['X'], self.data['y'])

    return tuned_model

  def __return_default_results(self):
    default_model = self.model_class()

    base_results = cross_validate(default_model, self.data['X'], self.data['y'], cv = self.cv_split, scoring=['r2', 'neg_mean_squared_error'], return_train_score=True)

    default_model.fit(self.data['X'], self.data['y']) #Just for defaults params

    data = {
      'params' : default_model.get_params(),
      'train mse' : base_results["train_neg_mean_squared_error"].mean(),
      'test mse': base_results["test_neg_mean_squared_error"].mean(),
      'train r2': base_results["train_r2"].mean(),
      'test r2': base_results["test_r2"].mean()
    }

    return data