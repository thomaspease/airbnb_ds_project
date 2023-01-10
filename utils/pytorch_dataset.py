from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale(df):
  scaler = MinMaxScaler()
  # I'm not sure why you need the iloc on the right hand side here, but you do...
  df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1])
  return df

class AirbnbNightlyPriceImageDataset(Dataset):
  def __init__(self, normalize=True):
    complete_df = pd.read_csv('/Users/tompease/Documents/Coding/airbnb/data/tabular_data/clean_tabular_data.csv')
    self.numeric_df = complete_df.select_dtypes(['number'])
    self.normalize = normalize

  def __getitem__(self, index):
    if self.normalize == True:
      df = scale(self.numeric_df)
      df = pd.DataFrame(df)
    else:
      df = self.numeric_df
    
    series = df.iloc[index]
    # These have to be cast as 32 bit tensors as the default is 64, but the default weights for pytorch linear layers are float32s, and they have to match...
    features = torch.tensor(series.drop(['Price_Night']), dtype=torch.float32)
    label = torch.tensor(series.loc[['Price_Night']], dtype=torch.float32)

    return (features, label)

  def __len__(self):
    return len(self.numeric_df)

dataset = AirbnbNightlyPriceImageDataset(normalize=True)

split_datasets = {}
split_datasets['train'], split_datasets['val'], split_datasets['test']= random_split(dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

data_loaders = {}
data_loaders['train'] = DataLoader(split_datasets['train'], batch_size=8, shuffle=True)
data_loaders['val'] = DataLoader(split_datasets['val'], batch_size=8, shuffle=True)
data_loaders['test'] = DataLoader(split_datasets['test'], batch_size=8, shuffle=True)