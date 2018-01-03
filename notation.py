import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
input will be one image with dimensions 5x5 and its an rgb cmap so,
it will have 3 channels, in this case the dim of input will be (1,5,5,3)

for demostration we will have only one filter and filter is of shape 3x3,
so dim of the kernel will be (3,3,3,1)

Output will depend on the strides, padding and filter size.
in this case stride will be 1 in all dimensions.
padding will be SAME(activation map dim will be same as input)

output dim order will be same as input dim order NHWC
N - batch size
H - Height of the image
W - Width of image
C - no.of filters
'''

a = np.arange(0,1*3*5*5).reshape(1,5,5,3)
fil = [[1,0,-1]]
fil = np.repeat(fil,9,axis = 0)
fil = fil.reshape(3,3,3,1)
print("Input image\n",a)

gph = tf.Graph()
with gph.as_default():
    x = tf.placeholder('float',shape = (1,5,5,3))
    kernel = tf.placeholder('float',shape =(3,3,3,1))
    out_img = tf.nn.conv2d(x,kernel,strides = [1,1,1,1],padding = 'SAME')
    out_shape = tf.shape(out_img)

with tf.Session(graph=gph) as sess:
    sess.run(tf.global_variables_initializer())
    output,output_shape = sess.run([out_img,out_shape],feed_dict={x:a,kernel:fil})
    print("\n----Activation map------\n",output,"\n--dim of activation map----\n",output_shape)

