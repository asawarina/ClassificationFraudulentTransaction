import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#df = pd.read_csv("creditcard.csv")
df= pd.read_csv("C:/Users/asawari/Desktop/BackUp1/buttonpython/media/creditcard.csv")

print(df.head(5))

def prep_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Convert the DataFrame into two variable
    X: data columns (V1 - V28)
    y: lable column
    """
    X = df.iloc[:, 2:30].values
    y = df.Class.values
    return X, y

def plot_data(X: np.ndarray, y: np.ndarray):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()

X, y = prep_data(df)

#plot_data(X, y)

print(pd.value_counts(pd.Series(y)))

def compare_plot(X: np.ndarray, y: np.ndarray, X_resampled: np.ndarray, y_resampled: np.ndarray, method: str):
      plt.subplot(1, 2, 1)
      plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
      plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
      plt.title('Original Set')
      plt.subplot(1, 2, 2)
      plt.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
      plt.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
      plt.title(method)
      plt.legend()
      #return plt.show()
      return plt.savefig('C:/Users/asawari/Desktop/BackUp1/buttonpython/media/smote.png')

 
method = SMOTE()
X_resampled, y_resampled =  method.fit_sample(X, y)
 
compare_plot(X, y, X_resampled, y_resampled, method= 'SMOTE') 
