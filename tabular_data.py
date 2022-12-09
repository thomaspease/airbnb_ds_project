import pandas as pd
import ast



def remove_rows_with_missing_values(df, columns):
  df = df.dropna(subset=columns)
  return df

def combine_description_strings(val):
  try:
    original_string = ast.literal_eval(val)
    listified_string = list(filter(lambda a: a != '', original_string))
    listified_string.remove('About this space')
    new_string = ' '.join(listified_string)
    new_string = new_string.replace('\n', ' ')
    return new_string
  except:
    if isinstance(val, str):
      return val
  
def set_default_feature_values(df):
  for val in ['beds', 'bathrooms', 'bedrooms', 'guests']:
    df[val].fillna(1)
  return df

def clean_tabular_data(df):
  df = remove_rows_with_missing_values(df, ['Cleanliness_rating', 'Description'])
  df.loc[:, 'Description'] = df.Description.apply(combine_description_strings)
  return df

def process_tabular_data():
  df = pd.read_csv('data/tabular_data/listing.csv')
  df = clean_tabular_data(df)
  df = set_default_feature_values(df)
  df.to_csv('data/tabular_data/clean_tabular_data.csv', index = False)

def load_airbnb(label):

  pass

if __name__ == "__main__":
  pass