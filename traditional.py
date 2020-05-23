# import sys
# from PIL import Image
# import base64

# image_fullpath=sys.argv[1]
# image_name=sys.argv[2]

# img= Image.open(str(image_fullpath))
# image_save_path=image_fullpath.replace(image_name,"temp.png")
# img.rotate(90).convert("LA").save(image_save_path)

# # base64_message = "/media/temp.png"
# print("/media/temp.png")
# # imsg = base64_message.encode('ascii')
# # msg_bytes = base64.b64decode(imsg)
# # mes = msg_bytes.decode('ascii')
# # print(mes)

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt



#df= pd.read_csv("creditcard.csv")
df= pd.read_csv("C:/Users/asawari/Desktop/BackUp1/buttonpython/media/creditcard.csv")




# Count the occurrences of fraud and no fraud and print them
occ = df['Class'].value_counts()
#print(occ)

# Print the ratio of fraud cases
print(occ / len(df.index))

#######ORIGINAL DATA#######

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
    #return plt.show()
    return plt.savefig('C:/Users/asawari/Desktop/BackUp1/buttonpython/media/traditional.png')

    

X, y = prep_data(df)

plot_data(X, y)
