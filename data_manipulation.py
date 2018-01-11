import pandas as pd
import numpy as np

labels = 0

def get_xdata(data_path):
    global labels
    df = pd.read_csv(data_path, sep=',', header=None)
    labels = np.array(df[df.columns[0]])
    df.drop(df.columns[0], axis=1, inplace=True)
    df.to_csv("x_data_train.csv",sep = ',',index = False)

def get_ydata():
    global y_train
    a = np.array(labels)
    b = np.zeros((len(labels), 10), dtype=np.int)
    b[np.arange(len(labels)), a] = 1
    y_data = np.array(b).astype(int)
    np.savetxt("y_data_train.csv", y_data, delimiter=",")


def main():
    get_xdata(data_path = "data/mnist_train.csv")
    get_ydata()


main()