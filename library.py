import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import f1_score  
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  #fill in the rest below

  # Define fit, but don't have it do anything
  def fit(self, X, y = None):
    print(f'\nWarning: {self.__class__.__name__}.fit does nothing.\n')
    return self

  # Write transform
  def transform(self, X):
    # Make sure X is a dataframe
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert (self.target_column in X.columns.to_list()), f'{self.__class__.__name__}.transform: \'{self.target_column}\' is not a column in the given DataFrame.'

    # One Hot Encoding
    X_ = pd.get_dummies(X,
                        prefix=self.target_column,
                        prefix_sep='_',
                        columns=[f'{self.target_column}'],
                        dummy_na=self.dummy_na,
                        drop_first=self.drop_first,
                      )
    return X_

  # Write fit_transform skipping fit
  def fit_transform(self, X, y = None):
    # self.fit(X, y)
    result = self.transform(X)
    return result


class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below
  def __init__(self, rename_dict:dict):
    assert isinstance(rename_dict, dict), f'{self.__class__.__name__} constructor expects a dict but got {type(rename_dict)} instead.'
    self.rename_dict = rename_dict

  #define fit to do nothing but give warning
  def fit(self, X, y = None):
    print(f'\nWarning: {self.__class__.__name__}.fit does nothing.\n')
    return self

  #write the transform method with asserts. Again, maybe copy and paste from MappingTransformer and fix up.
  def transform(self, X):
    # Make sure X is a dataframe
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    keys = self.rename_dict.keys()
    columns = X.columns.to_list()

    unknown_keys = keys - columns
    if unknown_keys:
      print(f'\nWarning: Unknown column name(s) found: {unknown_keys}\n')

    # Do the renaming
    X_ = X.copy()
    X_.rename(columns = self.rename_dict, inplace=True)
    return X_

  #write fit_transform that skips fit
  def fit_transform(self, X, y = None):
    # self.fit(X, y)
    result = self.transform(X)
    return result


class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result


class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence
    self.low = None
    self.high = None

  def fit(self, df, y = None):
    assert self.target_column in df, f'{self.target_column} is not a column in the DataFrame'
    """
    iqr = q3-q1  #inter-quartile range, where q1 is 25% and q3 is 75%
    inner_low = q1-1.5*iqr
    inner_high = q3+1.5*iqr
    For the outer fences:

    iqr = q3-q1  #inter-quartile range, where q1 is 25% and q3 is 75%
    outer_low = q1-3*iqr  #factor of 2 larger
    outer_high = q3+3*iqr
    """
    q3 = df[self.target_column].quantile(0.75)
    q1 = df[self.target_column].quantile(0.25)
    iqr = q3 - q1

    if self.fence == 'inner':
      self.low = q1 - 1.5 * iqr
      self.high = q3 + 1.5 * iqr
    else:
      self.low = q1 - 3 * iqr
      self.high = q3 + 3 * iqr

    return self


  def transform(self, df):
    assert self.low is not None or self.high is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    df_ = df.copy()
    df_[self.target_column] = df_[self.target_column].clip(lower=self.low, upper=self.high)
    return df_


  def fit_transform(self, df, y = None):
    self.fit(df, y)
    result = self.transform(df)
    return result


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.target_column = target_column
    self.minb = None
    self.maxb = None

  def fit(self, df, y = None):
    assert self.target_column in df, f'{self.target_column} is not a column in the DataFrame'

    mean = df[self.target_column].mean()
    sigma = df[self.target_column].std()

    self.maxb = mean + 3 * sigma
    self.minb = mean - 3 * sigma

    return self

  def transform(self, df):
    assert self.minb is not None or self.maxb is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    df_ = df.copy()
    df_[self.target_column] = df_[self.target_column].clip(lower=self.minb, upper=self.maxb)
    return df_

  def fit_transform(self, df, y = None):
    self.fit(df, y)
    result = self.transform(df)
    return result


class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    #fill in rest below
    self.column = column
    self.iqr = None
    self.median = None

  def fit(self, df, y = None):
    assert self.column in df, f'{self.column} is not a column in this {self.__class__.__name__} instance.'
    
    self.iqr = df[self.column].quantile(.75) - df[self.column].quantile(.25)
    self.median = df[self.column].median()
    
    return self

  def transform(self, df):
    assert self.iqr is not None or self.median is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    df_ = df.copy()
    df_[self.column] -= self.median
    df_[self.column] /= self.iqr

    return df_

  def fit_transform(self, df, y = None):
    self.fit(df, y)
    result = self.transform(df)
    return result


def find_random_state(features_df, labels, n=200):
  var = []  #collect test_error/train_error where error based on F1 score
  model = KNeighborsClassifier(n_neighbors=5)  #instantiate with k=5.
  
  #2 minutes
  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)           #predict against training set
      test_pred = model.predict(test_X)             #predict against test set
      train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
      test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
      f1_ratio = test_f1/train_f1          #take the ratio
      var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value

  rs = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return rs


titanic_variance_based_split = 107
customer_variance_based_split = 113

# Pipelines
titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', ce.TargetEncoder(cols=['Joined'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')),  #from chapter 5
    ('scale_fare', CustomRobustTransformer('Fare')),  #from chapter 5
    ('imputer', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False))  #from chapter 6
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', ce.TargetEncoder(cols=['ISP'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer('Age')), #from 5
    ('scale_time spent', CustomRobustTransformer('Time Spent')), #from 5
    ('impute', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False)),
    ], verbose=True)


def dataset_setup(original_table, label_column_name:str, the_transformer, rs, ts=.2):
  features = original_table.drop(columns=label_column_name) 
  labels = original_table[label_column_name].to_list()

  x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  
  x_train_transformed = the_transformer.fit_transform(x_train, y_train)
  x_test_transformed = the_transformer.transform(x_test)

  x_train_numpy = x_train_transformed.to_numpy()
  x_test_numpy = x_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_train_numpy, x_test_numpy, y_train_numpy, y_test_numpy


def titanic_setup(titanic_table, transformer=titanic_transformer, rs=titanic_variance_based_split, ts=.2):
  return dataset_setup(titanic_table, 'Survived', transformer, rs=rs, ts=ts)


def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
  return dataset_setup(customer_table, 'Rating', transformer, rs=rs, ts=ts)


def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy'])
  for t in thresh_list:
    yhat = [1 if v >=t else 0 for v in predicted]
    #note: where TP=0, the Precision and Recall both become 0. And I am saying return 0 in that case.
    precision = precision_score(actuals, yhat, zero_division=0)
    recall = recall_score(actuals, yhat, zero_division=0)
    f1 = f1_score(actuals, yhat)
    accuracy = accuracy_score(actuals, yhat)
    result_df.loc[len(result_df)] = {'threshold':t, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy}

  result_df = result_df.round(2)

  #Next bit fancies up table for printing. See https://betterdatascience.com/style-pandas-dataframes/
  #Note that fancy_df is not really a dataframe. More like a printable object.
  headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #800000; color: white; text-align: center"
  }
  properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

  fancy_df = result_df.style.format(precision=2).set_properties(**properties).set_table_styles([headers])
  return (result_df, fancy_df)



