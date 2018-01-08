import tensorflow as tf

gph = tf.Graph()
with gph.as_default():
    x = tf.placeholder('float',shape = [1],name = 'x_data')
    y = tf.placeholder('float',shape = [1],name = 'y_data')

    w = tf.Variable([5.0],name = 'Weight')

    o_pred = tf.multiply(x,w)
    cost = (y-o_pred)**2
    opt = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost,name = 'opt_op')

    saver = tf.train.Saver()

with tf.Session(graph=gph) as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(20):
        _,weight,loss = sess.run([opt,w,cost],feed_dict={x:[2],y:[4]})
        print("Cost: ",loss,"\tWeight: ",weight)

    saver.save(sess,'epochs/model')


