from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_loader import AirbnbLoader

def scale(df):
  scaler = MinMaxScaler()
  # I'm not sure why you need the iloc on the right hand side here, but you do...
  df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1])
  return df

class AirbnbPytorchDataset(Dataset):
  def __init__(self, label, normalize=True):
    data = AirbnbLoader()
    self.X, self.y = data.load_airbnb(label, normalize)

  def __getitem__(self, index):
    # These have to be cast as 32 bit tensors as the default is 64, but the default weights for pytorch linear layers are float32s, and they have to match...
    features = torch.tensor(self.X[index], dtype=torch.float32)
    label = torch.tensor(self.y[index], dtype=torch.float32)

    return (features, label)

  def __len__(self):
    return len(self.X)

dataset = AirbnbPytorchDataset('Price_Night', normalize=True)

split_datasets = {}
split_datasets['train'], split_datasets['val'] = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

data_loaders = {}
data_loaders['train'] = DataLoader(split_datasets['train'], batch_size=8, shuffle=True)
data_loaders['val'] = DataLoader(split_datasets['val'], batch_size=8, shuffle=True)