import torch
from utils.pytorch_dataset import data_loaders
import yaml
import optuna
from torch.utils.tensorboard import SummaryWriter

IN_FEATURES = 9
CLASSES = 2
EPOCHS = 10


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
  # layers.append(torch.nn.LogSoftMax(dim=1))

  return torch.nn.Sequential(*layers)

def train(trial):
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

      # Option to send to device
      # data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

      optimizer.zero_grad()
      prediction = model(data)
      loss = torch.nn.functional.mse_loss(prediction, target)
      loss.backward()
      optimizer.step()
    
    # Validation
    model.eval()
    loss = 0
    # Maybe need to add with torch.no_grad() here
    for batch_idx, (data, target) in enumerate(data_loaders['val']):
      prediction = model(data)
      loss += torch.nn.functional.mse_loss(prediction, target)

    trial.report(loss, epoch)

    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()

  return loss

if __name__ == "__main__":
  study = optuna.create_study(direction='minimize')
  study.optimize(train, n_trials=100, timeout=600)

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
      print("    {}: {}".format(key, value))





# class LinearRegression(torch.nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.linear_layer = torch.nn.Linear(9,1)

#   def forward(self, features):
#     return self.linear_layer(features)




# def train(config, epochs=100):
#   model = LinearRegression()
  
#   optimiser = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

#   writer = SummaryWriter()

#   batch_idx = 0

#   for epoch in range(epochs):
#     for phase in ['train', 'val']:
#       if phase == 'train':
#           model.train()  # Set model to training mode
#       else:
#           model.eval()

#       for batch in data_loaders[phase]:
#         optimiser.zero_grad()
#         features, labels = batch
#         prediction = model(features)
#         loss = torch.nn.functional.mse_loss(prediction, labels)
        
#         if phase == 'train':
#           loss.backward()
#           optimiser.step()
#           writer.add_scalar('loss', loss.item(), batch_idx)
        
#         else:
#           tune.report(loss.item())
#           writer.add_scalar('val loss', loss.item(), batch_idx)
#           batch_idx += 1
        

# with open("nn_config.yaml", "r") as stream:
#   config = (yaml.safe_load(stream))

# search_space = {
#     "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
#     "momentum": tune.uniform(0.1, 0.9),
# }



# tuner = tune.Tuner(train, param_space=search_space)
# results = tuner.fit()

# print(results.get_best_result(metric="score", mode="min").config)