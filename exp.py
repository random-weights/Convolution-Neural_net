import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x_path = "data/x_train.csv"
y_path = "data/y_train.csv"

df_x = pd.read_csv(x_path,sep = ',',header = None)
df_y = pd.read_csv(y_path,sep = ',',header = None)

x_data = np.array(df_x).reshape(len(df_x),28,28)
y_data = np.array(df_y).astype(np.int)

print(y_data[4])
plt.imshow(x_data[4],cmap = 'gray')
plt.show()
