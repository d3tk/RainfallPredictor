import warnings
from mimetypes import init

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout


def main():
    warnings.filterwarnings('ignore')

    weather_df = pd.read_csv(
        "data/train.csv")
    print(weather_df.shape)
    print(weather_df.head())
    print(weather_df.info())
    print(weather_df.isnull().sum())

    predict_df = pd.read_csv("data/test.csv")
    print(predict_df.shape)
    print(predict_df.head())
    print(predict_df.info())
    print(unknown_weather_df.isnull().sum())
    
if __name__ == "__main__":
    main()
