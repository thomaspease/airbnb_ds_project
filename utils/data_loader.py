import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale(df):
  scaler = MinMaxScaler()
  # I'm not sure why you need the iloc on the right hand side here, but you do...
  df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1])
  return df

class AirbnbLoader():
  def __init__(self):
    self.complete_df = pd.read_csv('/Users/tompease/Documents/Coding/airbnb/data/tabular_data/clean_tabular_data.csv')
    self.numeric_df = self.complete_df.select_dtypes(['number'])

  def load_airbnb(self, label, normalized=True):
    if normalized == True:
      df = scale(self.numeric_df)
      df = pd.DataFrame(df)
    else:
      df = self.numeric_df
    features_df = df.drop([label], axis=1)
    label_df = df.loc[:, label]
    return features_df, label_df
