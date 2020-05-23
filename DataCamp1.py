# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import pandas and read csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
#from PIL import Image

#from imblearn.over_sampling import RandomOverSampler
#from imblearn.over_sampling import SMOTE
#from sklearn.linear_model import LinearRegression, LogisticRegression
# =============================================================================
# from flask import Flask, redirect, url_for, render_template
# 
# app = Flask(__name__)
# 
# @app.route('/')
# def home():
#     return render_template('index.html')
# 
# if __name__ == '__main__':
#     app.run()
# =============================================================================
import sys 

output="Hi %s you're pretty" % (sys.argv[1])

print(output)

# # with open('Book1.json') as f:
# #     df = json.load(f)
# df = pd.read_csv("creditcard.csv")

# # # Explore the features available in your dataframe
# #print(df.info())
# #print(df)

# # html_code = df.get_html_string()
# # html_file = open('C:/Users/Guri/Desktop/_1AsawariProject/Anaconda/Django/buttonpython/templates/home.html','w')
# # html_file=html_file.write(html_code)
# # # Count the occurrences of fraud and no fraud and print them
# occ = df['Class'].value_counts()
# # #print(occ)

# # # Print the ratio of fraud cases
# print(occ / len(df.index))

# def prep_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
#     """
#     Convert the DataFrame into two variable
#     X: data columns (V1 - V28)
#     y: lable column
#     """
#     X = df.iloc[:, 2:30].values
#     y = df.Class.values
#     return X, y

# def plot_data(X: np.ndarray, y: np.ndarray):
#     plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
#     plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
#     plt.legend()
#     return plt.show()
# X, y = prep_data(df)
# plot_data(X, y)
# =============================================================================
# s = pd.Series([1, 2, 3])
# fig, ax = plt.subplots()
# s.plot.bar()
# fig.savefig('my_plot.png')
# =============================================================================

# =============================================================================

# h = Image(fig)
# h.show()
# =============================================================================



#####PYTHON'S IMBALANCE LEARN IS COMPATIBLE WITH SCIKIT WE IMPORT THE PACKAGE AND ASSIGN IT TO METHOD#######
#####fIT THE DATA SET X AND y AND THEN COMPARE#######

#plot_data(X_resampled,y_resampled)

#def compare_plot()
#compare_plot(X_resampled, y_resampled, X, y)

# =============================================================================
# method = RandomOverSampler()
# X_resampled, y_resampled =  method.fit_sample(X, y)
# 
# def compare_plot(X: np.ndarray, y: np.ndarray, X_resampled: np.ndarray, y_resampled: np.ndarray, method: str):
#     plt.subplot(1, 2, 1)
#     plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
#     plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
#     plt.title('Original Set')
#     plt.subplot(1, 2, 2)
#     plt.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
#     plt.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
#     plt.title(method)
#     plt.legend()
#     return plt.show()
#     
# 
#     
# compare_plot(X, y, X_resampled, y_resampled, method= 'ROS')
# =============================================================================

# =============================================================================
# method = SMOTE()
# X_resampled, y_resampled =  method.fit_sample(X, y)
# =============================================================================

#compare_plot(X, y, X_resampled, y_resampled, method= 'SMOTE')