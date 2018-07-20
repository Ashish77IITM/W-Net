from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from os.path import exists
from encoder import encode
from decoder import decode
from input_data import input_data
from soft_n_cut_loss import soft_n_cut_loss
import coloredlogs

logdir = "checkpoints/logs"
checkpt_dir_ckpt = "checkpoints/trained.ckpt"
checkpt_dir = "checkpoints"
num_classes = 2

# network parameters
learning_rate = 0.0001
num_steps = 100000000
display_step = 10
IMG_ROWS = 224
IMG_COLS = 224
global_step = 0

X = tf.placeholder(tf.float32, [None, IMG_ROWS, IMG_COLS, 3])

with tf.name_scope("Encoding"):
    encoded_image = encode(X)

with tf.name_scope("Decoding"):
    decoded_image = decode(encoded_image)

with tf.name_scope("Loss"):
    y_pred = tf.reshape(decoded_image, [-1, 150528])
    y_true = tf.reshape(X, [-1, 150528])
    soft_loss = soft_n_cut_loss(y_true, encoded_image, num_classes, IMG_ROWS, IMG_COLS)
    reconstruction_loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2)) 
    # print (X.get_shape())
    # print (decoded_image.get_shape())
    # img_concat = tf.concat(0, (tf.cast(X, tf.int32), tf.cast(decoded_image, tf.int32)), name = "Tensorboard_Images")

tf.summary.image("Input_image", X)
tf.summary.image("Ouptut_image",decoded_image)
tf.summary.scalar("SEE_loss", reconstruction_loss)
# tf.summary.scalar("Soft_Loss", soft_loss)
with tf.name_scope("Optimization"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    soft_op = optimizer.minimize(loss=soft_loss)
    recons_op = optimizer.minimize(loss=reconstruction_loss)

merged_summary = tf.summary.merge_all()

init = tf.global_variables_initializer()

saver = tf.train.Saver()
coloredlogs.install(level='DEBUG')
tf.logging.set_verbosity(tf.logging.DEBUG)

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    # print ("all is we")
    # print (tf.train.latest_checkpoint(checkpt_dir))
    # tf.logging.info('All is Well')
    if exists(checkpt_dir):
        # print ("\n\n\n\n\n\n\n\n\n")
        # print ("n\n\n\n\n\n\n\n\n\n")
        if tf.train.latest_checkpoint(checkpt_dir) is not None:
            tf.logging.info('Loading Checkpoint from '+ tf.train.latest_checkpoint(checkpt_dir))
            saver.restore(sess, tf.train.latest_checkpoint(checkpt_dir))
    iterator = input_data()
    next_items = iterator.get_next()

    for i in range(num_steps + 1):
        batch_x =  sess.run(next_items)
        _ = sess.run(soft_op, feed_dict={X: batch_x})
        _ = sess.run(recons_op, feed_dict={X: batch_x})

        if i % display_step == 0:
            recons_loss,soft_nloss, summary = sess.run([reconstruction_loss,soft_loss, merged_summary], feed_dict={X: batch_x})
            # recons_loss, summary = sess.run([reconstruction_loss, merged_summary], feed_dict={X: batch_x})
            tf.logging.info("Iteration number: ", str(tf.train.get_global_step()), "Recons Loss: ", str(recons_loss), "Soft-ncut", str(soft_nloss))
            # tf.logging.info('Iteration number: '+ str(i)+ " Recons Loss: "+ str(recons_loss)+ " Soft-ncut"+ str(0))
            train_writer.add_summary(summary)
            saver.save(sess, checkpt_dir_ckpt, global_step=tf.train.get_global_step())






