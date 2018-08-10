from __future__ import print_function
import tensorflow as tf

def LeNet(x, dropout):
    x = tf.reshape(x, [-1, 28, 28, 1])
    print('I/P shape ',x.get_shape())
  
    with tf.variable_scope('layer_A'):
        x = tf.layers.conv2d(x,filters=64,kernel_size=5, padding='SAME', activation=tf.nn.relu,                       
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print('Conv2D_A O/P shape ',x.get_shape())
        x = tf.layers.max_pooling2d(x, pool_size=(2,2), strides=2) # [-1, 14, 14, 64]
        print('MaxPool2D_A O/P shape ',x.get_shape())
        x = tf.layers.batch_normalization(x)
        print('BatchNormalization_A O/P shape ',x.get_shape())
        
    with tf.variable_scope('layer_B'):
        x = tf.layers.conv2d(x,filters=128,kernel_size=2, padding='SAME', activation=tf.nn.relu,                       
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print('Conv2D_B O/P shape ',x.get_shape())
        x = tf.layers.max_pooling2d(x, pool_size=(2,2), strides=2) # # [-1, 7, 7, 128]
        print('MaxPool2D_B O/P shape ',x.get_shape())
    
    with tf.variable_scope('layer_C'):
        x = tf.reshape(x, [-1, 7*7*128])
        print('FLATEN_C O/P shape ',x.get_shape())
        x = tf.layers.dense(x, 1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('DENSE_C O/P shape ',x.get_shape())
        x = tf.nn.dropout(x, keep_prob=1 - dropout)
        print('DROPOUT_C O/P shape ',x.get_shape())
      
    with tf.variable_scope('layer_D'):
        x = tf.layers.dense(x, 512, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('DENSE_D O/P shape ',x.get_shape())
        x = tf.layers.batch_normalization(x)
        print('BatchNormalization_D O/P shape ',x.get_shape())
    
    with tf.variable_scope('layer_E'):
        x = tf.layers.dense(x, 128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('DENSE_E O/P shape ',x.get_shape())
        
    with tf.variable_scope('layer_F'):
        x = tf.layers.dense(x, 47, kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('SOFTMAX_F O/P shape ',x.get_shape())
    
    return x