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


'''
Dimensions will be calculated manually since the no.of layers are small
but when the layers grow i will define new functions to populate the filter sizes 
and compute FC layer dimensions.

graph looks like:

Input -> conv1 -> pool1 -> relu-> conv2 -> pool2 -> relu-> conv3 -> pool3 -> relu

-> conv4 -> pool4 -> relu -> FC1 -> RelU -> FC2 -> Softmax 

'''

epochs = 1000
batch_size = 128

gph = tf.Graph()
with gph.as_default():

    #define placeholders
    x = tf.placeholder('float',shape = (batch_size,28,28,1))
    y = tf.placeholder('float',shape = (batch_size,10))

    initializer = tf.contrib.layers.xavier_initializer()
    kern = [0]*4
    bias = [0]*4

    ls_filt_dims = [[3,3,1,16],[3,3,16,32],[3,3,32,64],[3,3,64,128]]
    fmap = x
    for i in range(4):
        kern[i] = tf.Variable(initializer(shape = ls_filt_dims[i]))
        bias[i] = tf.Variable(initializer(shape = [ls_filt_dims[i][3]]))
        fmap = tf.nn.conv2d(fmap,kern[i],[1,1,1,1],'VALID')
        layer = tf.nn.relu(fmap + bias[i])
        layer_pool = tf.nn.max_pool(layer,[1,2,2,1],[1,1,1,1],'VALID')
        fmap = layer_pool

    #the list out_dims is calculated manually
    out_dims = [batch_size,16,16,128]
    flat_dim = out_dims[1]*out_dims[2]*out_dims[3]

    # fully connected layer
    fw = [0]*4
    fb = [0]*4

    ls_weight_dims = [[flat_dim, 128], [128, 64], [64, 32], [32, 10]]
    fa = tf.reshape(fmap,shape = (out_dims[0],flat_dim))
    for i in range(4):
        fw[i] = tf.Variable(initializer(shape = ls_weight_dims[i]))
        fb[i] = tf.Variable(initializer(shape = [ls_weight_dims[i][1]]))
        fz = tf.matmul(fa,fw[i])+fb[i]
        fa = tf.nn.relu(fz)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fz,labels=y))
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



