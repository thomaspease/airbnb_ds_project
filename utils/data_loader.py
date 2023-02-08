import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def scale(df):
  scaler = MinMaxScaler()
  # I'm not sure why you need the iloc on the right hand side here, but you do...
  df.iloc[:,:] = scaler.fit_transform(df.iloc[:,:])
  return df

class AirbnbLoader():
  def __init__(self):
    self.complete_df = pd.read_csv('/Users/tompease/Documents/Coding/airbnb/data/tabular_data/clean_tabular_data.csv')
    self.numeric_df = self.complete_df.select_dtypes(['number'])
    self.X_test = None
    self.y_test = None

  def load_airbnb(self, label, normalized=True):
    if normalized == True:
      df = scale(self.numeric_df)
      df = pd.DataFrame(df)
    else:
      df = self.numeric_df
    
    # This is because at this point df is just numeric so if you have the category as the label then it won't be in df
    if label in df.columns:
      features_df = df.drop([label], axis=1)
      label_df = df.loc[:, label]
    else:
      features_df = df
      label_df = self.complete_df.loc[:, label]

    # This stores a holdout set in the loader which should be the same every time
    X_train, self.X_test, y_train, self.y_test = train_test_split(features_df, label_df, test_size=0.15, random_state=42)
    
    return X_train, y_train

  def load_test_sets(self, label):
    self.load_airbnb(label)
    return self.X_test, self.y_test