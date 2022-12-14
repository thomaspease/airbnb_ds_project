import pandas as pd

class DataCleaner():
  '''
  A class used to clean csv data

  Attributes:
    init_df (pandas dataframe): the initial data as loaded from a csv file
  '''
  def __init__(self, csv_path):
    self.init_df = pd.read_csv(csv_path)

  def remove_rows_with_missing_values(self, df, columns):
    df = df.dropna(subset=columns)
    return df

  def set_default_feature_values(self, df, columns):
    for val in columns:
      df[val] = df[val].fillna(1)
    return df

class AirbnbDataCleaner(DataCleaner):
  '''
  A class containing methods specifically for cleaning data on the Airbnb property dataset

  Attributes:
    See help(DataCleaner) for attributes
  '''
  def combine_description_strings(self, orig_str):
    '''
    A function which cleans up the property description. In it's raw form it is a string written with python list syntax. It is outputted as a fairly clean string.
    '''
    try:
      listified_str = ast.literal_eval(orig_str)
      listified_str = list(filter(lambda a: a != '', listified_str)) # Removes the empty strings from the list
      listified_str.remove('About this space')
      new_string = ' '.join(listified_str) #Joins the list back together into a string
      new_string = new_string.replace('\n', ' ')
      return new_string
    # There are a few descriptions which are already a short string and don't need processing
    except:
      if isinstance(orig_str, str):
        return orig_str
    
  def clean_tabular_data(self):
    '''
    A function which sequentially calls the functions needed to clean the Airbnb tabular data
    '''
    df = self.init_df
    cleaned_df = df.drop(['Unnamed: 19'], axis=1)
    cleaned_df = self.remove_rows_with_missing_values(cleaned_df, ['Cleanliness_rating', 'Description'])
    cleaned_df.loc[:, 'Description'] = cleaned_df.Description.apply(self.combine_description_strings)
    cleaned_df = self.set_default_feature_values(df=cleaned_df, columns=['beds', 'bathrooms', 'bedrooms', 'guests'])
    cleaned_df.to_csv('/Users/tompease/Documents/Coding/airbnb/data/tabular_data/clean_tabular_data.csv', index = False)

if __name__ == "__main__":
  airbnb_cleaner = AirbnbDataCleaner('/Users/tompease/Documents/Coding/airbnb/data/tabular_data/listing.csv')
  airbnb_cleaner.clean_tabular_data()