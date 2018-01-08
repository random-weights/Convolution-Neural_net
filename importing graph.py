import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('epochs/model.meta')
new_saver.restore(sess,tf.train.latest_checkpoint('epochs/'))

graph = tf.get_default_graph()
x_data = graph.get_tensor_by_name("x_data:0")
y_data = graph.get_tensor_by_name("y_data:0")
feed_dict = {x_data:[2],y_data:[4]}

for i in range(10):
    sess.run('opt_op', feed_dict)
print(sess.run('Weight:0'))


