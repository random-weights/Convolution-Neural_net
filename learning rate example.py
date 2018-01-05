import tensorflow as tf


gph = tf.Graph()
with gph.as_default():
    x = tf.placeholder('float',shape = [1])
    y = tf.placeholder('float',shape = [1])

    w = tf.Variable(0.0)
    y_pred = tf.multiply(w,x)
    cost = (y-y_pred)**2

    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step)

with tf.Session(graph = gph) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _,loss = sess.run([opt,cost],feed_dict={x:[2.0],y:[4.0]})
        print("Epoch: ",i,"\tLoss: ",loss,"\tGlobal_step: ",global_step.eval(),"\tlearning_rate: ",learning_rate.eval())
