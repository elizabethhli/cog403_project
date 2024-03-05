# import all torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


import pandas as pd

# LOAD DATA
df_survey = pd.read_csv('data/SharedResponsesSurvey.csv')
print(f"Number of participants: ", df_survey.shape[0] // 26)

headers = df_survey.columns
print(f"Header names: ", headers)

# split test, valid, train datasets
TARGET = 'Saved'
X = df_survey.drop(TARGET, axis=1)
y = df_survey[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

# TODO: define model and train + validate + test
# if __name__ == "__main__":
    # print(df_survey.head())