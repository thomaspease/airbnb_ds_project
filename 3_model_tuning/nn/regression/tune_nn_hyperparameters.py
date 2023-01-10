import sys
sys.path.append('/Users/tompease/Documents/Coding/airbnb')
from utils.pytorch_dataset import data_loaders
import torch
import optuna
import os
from joblib import dump
import json
from datetime import datetime 
from torch.utils.tensorboard import SummaryWriter

# DEVICE = torch.device('cpu')
# BATCHSIZE = 
# N_TRAIN_EXAMPLES = BATCHSIZE * 30
# N_VALID_EXAMPLES = BATCHSIZE * 10
IN_FEATURES = 9
EPOCHS = 20

def define_model(trial):
  n_layers = trial.suggest_int("n_layers", 1, 3)
  layers = []

  in_features = IN_FEATURES

  for i in range(n_layers):
    out_features = trial.suggest_int(f"n_units_l{i}", 2, 18)
    layers.append(torch.nn.Linear(in_features, out_features))
    layers.append(torch.nn.ReLU())
    p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.8)
    layers.append(torch.nn.Dropout(p))

    in_features = out_features
  
  layers.append(torch.nn.Linear(in_features, 1))

  return torch.nn.Sequential(*layers)

class Objective():
  def __init__(self):
    self.best_loss = None
    self.best_model = None

  def __call__(self, trial):
    model = define_model(trial)

    optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    # The first half finds the relevant class of optimizer by using the names as passed in above, and then you pass in the parameters and the learning rate into that model
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    for epoch in range(EPOCHS):
      # Training of the model
      model.train()
      
      for batch_idx, (data, target) in enumerate(data_loaders['train']):
        # Option to limit data for faster epochs
        # if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
        #   break
        # Option to send to device if using GPU
        # data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        prediction = model(data)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
      
      # Validation
      model.eval()
      loss = 0
      with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loaders['val']):
          # Option to limit validation data.
          # if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
          #     break
          # Option to send to device if using GPU
          # data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
          
          prediction = model(data)
          loss += torch.nn.functional.mse_loss(prediction, target)

      trial.report(loss, epoch)

      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

      # Set best loss on first trial
      if self.best_loss == None:
        self.best_loss = loss.item()
      # Keep model if loss is best
      elif loss.item() < self.best_loss:
        self.best_model = model

    return loss

def save_model_and_results(objective, study):
  now = datetime.now()
  dt_string = now.strftime("%d%m%Y_%H:%M")

  path = f'/Users/tompease/Documents/Coding/airbnb/models/nns/regression/{objective.best_model.__class__.__name__}/{dt_string}'

  if not os.path.exists(path):
    os.makedirs(path)

  dump(objective.best_model, f'{path}/model.joblib')

  metrics = {
    'mse': study.best_trial.value,
    'params': dict(study.best_trial.params.items())
  }

  with open(f'{path}/metrics.json', "w") as outfile:
    json.dump(metrics, outfile, indent=4)

if __name__ == "__main__":
  study = optuna.create_study(direction='minimize')
  objective = Objective()
  study.optimize(objective, n_trials=100, timeout=600)

  pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
  complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

  print("Study statistics: ")
  print("  Number of finished trials: ", len(study.trials))
  print("  Number of pruned trials: ", len(pruned_trials))
  print("  Number of complete trials: ", len(complete_trials))

  print("Best trial:")
  trial = study.best_trial

  print("  Value: ", trial.value)

  print("  Params: ")
  for key, value in trial.params.items():
      print(f"    {key}: {value}")

  save_model_and_results(objective, study)