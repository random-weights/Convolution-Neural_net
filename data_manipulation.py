import pandas as pd
import numpy as np

labels = 0

def get_xdata(data_path):
    global labels
    df = pd.read_csv(data_path, sep=',', header=None)
    labels = np.array(df[df.columns[0]]).astype(np.int)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.to_csv("x_data_train.csv",sep = ',',index = False,header = None)

def get_ydata():
    a = np.array(labels)
    b = np.zeros((len(labels), 10), dtype=np.int)
    b[np.arange(len(labels)), a] = 1
    b = b.astype(int)
    y_data = pd.DataFrame(b,index = None,columns = None)
    y_data.to_csv("y_train.csv",sep = ',',index = False,header=None)


def main():
    get_xdata(data_path = "data/mnist_train.csv")
    get_ydata()


main()