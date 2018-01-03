import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class Data():

    def __init__(self,data_path = None):
        if data_path is None:
            self.data_path = "data/mnist_train.csv"
        else:
            self.data_path = data_path
        self.get_xdata()
        self.get_ydata()

    def get_xdata(self):
        df = pd.read_csv(self.data_path, sep=',', header=None)
        self.labels = np.array(df[df.columns[0]])
        df.drop(df.columns[0], axis=1, inplace=True)
        a = np.zeros(shape=(len(df), 28, 28))
        for i in range(len(df)):
            a[i] = df.iloc[i].values.reshape(28, 28)
        self.x_train = a

    def get_ydata(self,labels = None):

        if labels is None:
            temp_labels = self.labels
        else:
            temp_labels = labels

        a = np.array(temp_labels)
        b = np.zeros((len(temp_labels), 10), dtype=np.int)
        b[np.arange(len(temp_labels)), a] = 1
        self.y_train = np.array(b)
        return self.y_train

    def get_rand_batch(self,batch_size = None):
        if batch_size is None:
            b_size = 100
        else:
            b_size = batch_size

        rand_indices = np.random.choice(60000, b_size, replace=False)
        x_batch_train = self.x_train[rand_indices]
        self.x_batch_train = x_batch_train.reshape(b_size, 28, 28, 1)
        self.y_batch_train = self.y_train[rand_indices]


class CNN():

    def __init__(self,ls_window_size = None):
        self.ls_filters = [100,75,50,25]
        if ls_window_size is None:
            self.ls_win_size = [3,3,3,3]
        else:
            self.ls_win_size = ls_window_size

    def setInputDimensions(self,in_dimensions):
        self.in_dims = in_dimensions
        self.ls_filters = [self.in_dims[3]] + self.ls_filters
        self.fmap_dims = self.in_dims

    def getOutputDimensions(self):
        return self.fmap_dims

    def generateFilterDimensions(self):
        n_conv_layers = len(self.ls_filters) - 1
        ls_filt_dims = []
        for i in range(n_conv_layers):
            ls_filt_dims.append([self.ls_win_size[i],self.ls_win_size[i],self.ls_filters[i],self.ls_filters[i+1]])
        self.ls_filt_dims = ls_filt_dims
        return(self.ls_filt_dims)

    def layer(self,kernel_dims,stride,padding,Type):
        fmap = self.fmap_dims
        fmap[1] = int((fmap[1]+2*padding - kernel_dims[1])/stride[1] + 1)
        fmap[2] = fmap[1]
        if Type == 'CONV':
            fmap[3] = kernel_dims[3]

        self.fmap_dims = fmap


epochs = 1000
batch_size = 100

out_dims = 0
ls_filt_dims = []
def main():
    cnn1 = CNN()
    cnn1.setInputDimensions([100,28,28,1])
    global out_dims, ls_filt_dims
    ls_filt_dims = cnn1.generateFilterDimensions()
    cnn1.layer(ls_filt_dims[0], [1, 1, 1, 1], 0, Type='CONV')
    cnn1.layer([1, 2, 2, 1], [1, 1, 1, 1], 0, Type='POOL')
    cnn1.layer(ls_filt_dims[1], [1, 1, 1, 1], 0, Type='CONV')
    cnn1.layer([1, 2, 2, 1], [1, 1, 1, 1], 0, Type='POOL')
    cnn1.layer(ls_filt_dims[2], [1, 1, 1, 1], 0, Type='CONV')
    cnn1.layer([1, 2, 2, 1], [1, 1, 1, 1], 0, Type='POOL')
    cnn1.layer(ls_filt_dims[3], [1, 1, 1, 1], 0, Type='CONV')
    cnn1.layer([1, 2, 2, 1], [1, 1, 1, 1], 0, Type='POOL')
    out_dims = cnn1.getOutputDimensions()

main()


gph = tf.Graph()
with gph.as_default():

    #define placeholders
    x = tf.placeholder('float',shape = (batch_size,28,28,1))
    y = tf.placeholder('float',shape = (batch_size,10))

    # computing no.of input features for Fully connected layer
    initializer = tf.contrib.layers.xavier_initializer()
    kern = [0]*4
    bias = [0]*4

    fmap = x
    for i in range(4):
        kern[i] = tf.Variable(initializer(shape = ls_filt_dims[i]))
        bias[i] = tf.Variable(initializer(shape = [ls_filt_dims[i][3]]))
        fmap = tf.nn.conv2d(fmap,kern[i],[1,1,1,1],'VALID')
        layer = tf.nn.relu(fmap + bias[i])
        layer_pool = tf.nn.max_pool(layer,[1,2,2,1],[1,1,1,1],'VALID')
        fmap = layer_pool

    flat_dim = out_dims[1]*out_dims[2]*out_dims[3]

    # fully connected layer
    flayer_in = tf.reshape(fmap,shape = (out_dims[0],flat_dim))
    fw1 = tf.Variable(initializer(shape = (flat_dim,100)),name = "fw1")
    fb1 = tf.Variable(initializer(shape=[100]),name = "fb1")
    fw2 = tf.Variable(initializer(shape = (100,50)),name = "fw2")
    fb2 = tf.Variable(initializer(shape=[50]),name = "fb2")
    fw3 = tf.Variable(initializer(shape=(50,25)),name = "fw3")
    fb3 = tf.Variable(initializer(shape=[25]),name = "fb3")
    fw4 = tf.Variable(initializer(shape=(25,10)),name = "fw4")
    fb4 = tf.Variable(initializer(shape=[10]),name = "fb4")

    fz1 = tf.matmul(flayer_in,fw1) + fb1
    fa1 = tf.nn.relu(fz1)

    fz2 = tf.matmul(fa1, fw2) + fb2
    fa2 = tf.nn.relu(fz2)

    fz3 = tf.matmul(fa2, fw3) + fb3
    fa3 = tf.nn.relu(fz3)

    fz4 = tf.matmul(fa3, fw4) + fb4

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fz4,labels=y))
    opt = tf.train.AdamOptimizer().minimize(cost)

with tf.Session(graph=gph) as sess:
    sess.run(tf.global_variables_initializer())
    train_data = Data()

    ls_loss = []
    for i in range(epochs):
        train_data.get_rand_batch(batch_size)
        _,loss = sess.run([opt,cost],feed_dict={x:train_data.x_batch_train,y:train_data.y_batch_train})
        ls_loss.append(loss)
        print("epoch: ",i,"\tLoss: ",loss)

    plt.plot(range(epochs),ls_loss)
    plt.show()



