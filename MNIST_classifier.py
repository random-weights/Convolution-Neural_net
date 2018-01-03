import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

y_batch_train =0
x_batch_train = 0
y_train = 0
x_train = 0
labels = 0

def data():
    #print("Reading from CSV file..")
    df = pd.read_csv("data/mnist_train.csv",sep=',',header = None)
    global labels,x_train
    labels = np.array(df[df.columns[0]])
    df.drop(df.columns[0],axis = 1,inplace = True)
    a = np.zeros(shape = (len(df),28,28))
    #print("\nConverting to numpy array..")
    for i in range(len(df)):
        a[i] = df.iloc[i].values.reshape(28,28)

    x_train = a


def one_hot():
    a = np.array(labels)
    b = np.zeros((len(labels), 10), dtype=np.int)
    b[np.arange(len(labels)), a] = 1

    global y_train
    y_train = np.array(b)


def get_training_data(batch_size):

    rand_indices = np.random.choice(60000,batch_size,replace = False)
    #print("\nExtracting Random Values..")
    global y_batch_train,x_batch_train
    x_batch_train = x_train[rand_indices]
    x_batch_train = x_batch_train.reshape(batch_size,28,28,1)
    y_batch_train = y_train[rand_indices]


batch_size = 100
epochs = 1000


gph = tf.Graph()
with gph.as_default():

    #define placeholders
    x = tf.placeholder('float',shape = (batch_size,28,28,1))
    y = tf.placeholder('float',shape = (batch_size,10))

    initializer = tf.contrib.layers.xavier_initializer()
    # --------------layer 1------------------
    in_shape = tf.shape(x)
    kern1 = tf.Variable(initializer(shape =(3,3,1,30)),name = "kern1")
    map1 = tf.nn.conv2d(x,kern1,strides = [1,1,1,1],padding = 'VALID')
    map_shape = tf.shape(map1)
    bias1 = tf.Variable(initializer(shape = [30]),name = "bias1")
    l1 = tf.nn.relu(map1 + bias1)

    #---------pooling layer1------------------
    l1_pool = tf.nn.max_pool(l1,[1,2,2,1],strides = [1,1,1,1],padding = 'VALID')
    in_shape = tf.shape(l1_pool)

    #----------layer2---------------------
    kern2 = tf.Variable(initializer(shape = (3,3,30,20)),name = "kern2")
    map2 = tf.nn.conv2d(l1_pool,kern2,strides = [1,1,1,1],padding = 'VALID')
    map_shape = tf.shape(map2)
    bias2 = tf.Variable(initializer(shape = [20]),name = "bias2")
    l2 = tf.nn.relu(map2+bias2)

    # --------------pooling layer2---------------
    l2_pool = tf.nn.max_pool(l2,[1,2,2,1],strides = [1,1,1,1],padding = 'VALID')
    in_shape = tf.shape(l2_pool)

    # ---------------layer3---------------------
    kern3 = tf.Variable(initializer(shape=(3, 3,20, 10)),name = "kern3")
    map3 = tf.nn.conv2d(l2_pool, kern3, strides=[1, 1, 1, 1], padding='VALID')
    map_shape = tf.shape(map3)
    bias3 = tf.Variable(initializer(shape=[10]),name = "bias3")
    l3 = tf.nn.relu(map3 + bias3)

    # --------------pooling layer3---------------
    l3_pool = tf.nn.max_pool(l3, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
    in_shape = tf.shape(l3_pool)

    # ---------------layer4---------------------
    kern4 = tf.Variable(initializer(shape=(3, 3, 10, 5)),name = "kern4")
    map4 = tf.nn.conv2d(l3_pool, kern4, strides=[1, 1, 1, 1], padding='VALID')
    map_shape = tf.shape(map4)
    bias4 = tf.Variable(initializer(shape=[5]),name = "bias4")
    l4 = tf.nn.relu(map4 + bias4)

    # --------------pooling layer4---------------
    l4_pool = tf.nn.max_pool(l4, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
    in_shape = tf.shape(l4_pool)

    # fully connected layer
    flat_dim = in_shape[1]*in_shape[2]*in_shape[3]
    flayer_in = tf.reshape(l4_pool,shape = (in_shape[0],flat_dim))
    fw1 = tf.Variable(initializer(shape = (16*16*5,100)),name = "fw1")
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
    ls_loss = []
    data()
    one_hot()
    for i in range(epochs):
        get_training_data(batch_size)
        _,loss = sess.run([opt,cost],feed_dict={x:x_batch_train,y:y_batch_train})
        ls_loss.append(loss)
        print("epoch: ",i,"\tLoss: ",loss)

    plt.plot(range(epochs),ls_loss)
    plt.show()





