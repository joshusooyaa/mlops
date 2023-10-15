import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

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
