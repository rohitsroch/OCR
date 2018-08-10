from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from model import LeNet
from tqdm import tqdm

flags = tf.app.flags
flags.DEFINE_integer("epoch",20, "Epochs to train (Def: 20).")
flags.DEFINE_integer("batch_size",100, "Batch size (Def: 100).")
flags.DEFINE_string("save_path", "./model_ckpt", "Path to save out model files.")
flags.DEFINE_string("model_name", "LeNet", "name of model.")
flags.DEFINE_string("latest_ckpt_name", "LeNet_20", "name of latest model checkpoint.")
flags.DEFINE_float("learning_rate",1e-3, "learning_rate (Def: 0.001)")

FLAGS = flags.FLAGS

    
def computeLoss(ys, model_logits):
    """
      Cross Entropy loss

    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys,logits=model_logits)) 
    return loss


def trainModel(train_data , train_labels, test_data, test_labels):
    """Train for a number of steps."""
    tf.reset_default_graph()
    xs = tf.placeholder(tf.float32, [None, 784], name='input')
    ys = tf.placeholder(tf.float32, [None, 47], name='exp_output')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    #get the model logits
    model_logits = LeNet(xs , dropout)
    loss = computeLoss(ys, model_logits)
    
    #define optimizer
    train_opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    #define accracy
    correct = tf.equal(tf.argmax(model_logits, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    batch_size= FLAGS.batch_size
    num_examples_per_epoch = len(train_data)
    num_of_batches_per_epoch = (num_examples_per_epoch / batch_size)
    no_of_epoch_for_train= FLAGS.epoch
    save_path = FLAGS.save_path
    
    print('Batch-Size', batch_size)
    print('num_examples_per_epoch', num_examples_per_epoch)
    print('num_of_batches_per_epoch', num_of_batches_per_epoch)
    print('no_of_epoch_for_train', no_of_epoch_for_train)
    
    
    with tf.Session() as sess:
        sess.run(init)
                
        print('[*] Reading checkpoints...')
        saver.restore(sess, os.path.join(FLAGS.save_path,FLAGS.latest_ckpt_name))
        print('[*] Read {}'.format(FLAGS.latest_ckpt_name))

        curr_epoch = 20 
        losses = []
        for epoch in tqdm(range(no_of_epoch_for_train)):
            avg_loss = 0.
            for i in range(int(num_of_batches_per_epoch)):
                x_batches, y_batches = train_data[i * 100: (i + 1) * 100], train_labels[i * 100: (i + 1) * 100]
                _, loss_batch = sess.run([train_opt, loss], feed_dict={xs: x_batches, ys: y_batches, dropout: 0.5})
                avg_loss = avg_loss + loss_batch
            
            
            losses.append(avg_loss/num_of_batches_per_epoch)
            train_accuracy_per_epoch = sess.run(accuracy, feed_dict={xs: train_data,ys: train_labels,dropout: 0.5})
            print (('Train: Epoch %d, loss = %.5f, Accuracy = %.5f') % (curr_epoch, 
                                                                        avg_loss/num_of_batches_per_epoch, train_accuracy_per_epoch))
            test_accuracy_per_epoch = sess.run(accuracy, feed_dict={xs: test_data,ys: test_labels,dropout: 0.0})
            print(('Validation : Epoch %d, Accuracy = %.5f') % (curr_epoch, test_accuracy_per_epoch))
            saver.save(sess,os.path.join(FLAGS.save_path, FLAGS.model_name + '_'+ str(curr_epoch)))
            np.savetxt(os.path.join(FLAGS.save_path, 'losses.txt'), losses)
            curr_epoch = curr_epoch + 1
            
        print('Done Training!')
                 

            
            
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])               
    
def main(_): 
    train = pd.read_csv('./emnist-balanced-train.csv', header=None) #path for train-set EMNIST data
    test = pd.read_csv('./emnist-balanced-test.csv', header=None) #path for test-set EMNIST data
    #split image and label
    train_data = train.iloc[:, 1:]
    train_labels = train.iloc[:, 0]
    test_data = test.iloc[:, 1:]
    test_labels = test.iloc[:, 0]
    #one hot encode labels for train and test set
    train_labels = pd.get_dummies(train_labels)
    test_labels = pd.get_dummies(test_labels)
    #get numpy array for train and test
    train_data = train_data.values
    train_labels = train_labels.values
    test_data = test_data.values
    test_labels = test_labels.values
    
    train_data = np.apply_along_axis(rotate, 1, train_data)/255.
    test_data = np.apply_along_axis(rotate, 1, test_data)/255.
    print("Starting Model Training!")
    trainModel(train_data , train_labels, test_data, test_labels)
    

if __name__ == '__main__':
    tf.app.run()   
