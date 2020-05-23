import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from imblearn.pipeline import Pipeline 

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import precision_recall_curve, average_precision_score


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

#ACCURACY SCORE AND PRECISION RECALL CURVE USING RANDOM FOREST
# import the methods
# # Split your data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 
# # Define the model as the random forest
model = RandomForestClassifier(random_state=5, n_estimators=20)
# # Fit the model to our training set
model.fit(X_train, y_train)
# # Obtain predictions from the test data 
predicted = model.predict(X_test)

print(f'Accuracy Score:\n{accuracy_score(y_test, predicted):0.3f}')
# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, predicted)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, predicted)
print(f'Precision: {precision}\nRecall: {recall}')

def plot_pr_curve(recall, precision, average_precision):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    from inspect import signature
    plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(f'2-class Precision-Recall curve: AP={average_precision:0.2f}')
    #return plt.show()
    return plt.savefig('C:/Users/asawari/Desktop/BackUp1/buttonpython/media/prcurve.png')
# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)