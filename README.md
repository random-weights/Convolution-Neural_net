# Convolution-Neural_net
CNN for MNIST dataset

Master Branch contains the general implementation in tensorflow.\
layer-api branch contains the same network implemented using tf.layers api.

## Advantages of tf.layers api over tensorflow core
I no longer need to keep track of output dimensions after each convolution and pooling, 
api takes care of deciding shapes of weights for fully connected layer - 1
