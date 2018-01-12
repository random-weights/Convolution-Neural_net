import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class Data():
    def __init__(self):
        self.size = 0

    def get_xdata(self,x_data_path):
        df = pd.read_csv(x_data_path, sep=',', header=None)
        a = np.array(df).astype(int)
        self.size = len(df)
        a = a.reshape(self.size,28,28)
        self.x_data = a
        return self.x_data

    def get_ydata(self,y_data_path):
        df = pd.read_csv(y_data_path,sep = ',',header = None)
        b = np.array(df).astype(int)
        b = b.reshape(len(df),10)
        self.y_data = b
        return self.y_data

    def get_rand_batch(self,batch_size = None):
        if batch_size is None:
            b_size = 100
        else:
            b_size = batch_size

        rand_indices = np.random.choice(self.size, b_size, replace=False)
        x_batch = self.x_data[rand_indices]
        self.x_batch = x_batch.reshape(b_size, 28, 28, 1)
        self.y_batch = self.y_data[rand_indices]


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

start_learning = 0.01
# learning rate at the start of training

gph = tf.Graph()
with gph.as_default():

    # define placeholders
    x = tf.placeholder('float',shape = (None,28,28,1))
    y = tf.placeholder('float',shape = (None,10))

    initializer = tf.contrib.layers.xavier_initializer()
    kern = [0]*2
    bias = [0]*2

    ls_filt_dims = [[5,5,1,32],[5,5,32,64]]
    fmap = x
    for i in range(2):
        kern[i] = tf.Variable(initializer(shape = ls_filt_dims[i]))
        bias[i] = tf.Variable(initializer(shape = [ls_filt_dims[i][3]]))
        fmap = tf.nn.conv2d(fmap,kern[i],[1,1,1,1],'SAME')
        layer = tf.nn.relu(fmap + bias[i])
        layer_pool = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1],'SAME')
        fmap = layer_pool

    # the list out_dims is calculated manually
    out_dims = [tf.shape(x)[0],7,7,64]
    flat_dim = out_dims[1]*out_dims[2]*out_dims[3]

    # fully connected layer
    fw = [0]*2
    fb = [0]*2

    ls_weight_dims = [[flat_dim, 1024], [1024, 10]]
    fa = tf.reshape(fmap,shape = (out_dims[0],flat_dim))
    for i in range(2):
        fw[i] = tf.Variable(initializer(shape = ls_weight_dims[i]))
        fb[i] = tf.Variable(initializer(shape = [ls_weight_dims[i][1]]))
        fz = tf.matmul(fa,fw[i])+fb[i]
        fa = tf.nn.relu(fz)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fz,labels=y))

    # calculating accuracy
    fa = tf.nn.softmax(logits=fz)
    _,acc = tf.metrics.accuracy(tf.argmax(fa,axis = 1),tf.argmax(y,axis = 1))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning,global_step,100,0.96,staircase=True)
    # updates learning rate for every 100 epochs since we are doing batch training

    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
    saver = tf.train.Saver()


with tf.Session(graph=gph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_data = Data()
    train_data.get_xdata("data/x_train.csv")
    train_data.get_ydata("data/y_train.csv")

    i = 0
    count = 0
    while(i<epochs):
        i = i+1
        count += 1
        train_data.get_rand_batch(batch_size)
        _,train_acc = sess.run([opt,acc],feed_dict={x:train_data.x_batch,y:train_data.y_batch})
        print("epoch: ",i,"\tAccuracy: ",train_acc)

        if i == epochs-1:
            flag = input("Continue(y/n): ")
            if flag == 'y':
                i = 0
            else:
                break

    # saver.save(sess,"cnnmodel/model1")
    plt.show()