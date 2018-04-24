import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import genSmallTextImg
import tools


word, text = genSmallTextImg.genSmallTextImg()
letters = genSmallTextImg.byGradient(word)
n = len(letters)
image_size = 128
x_batch = []

for letter in letters:
	img = cv2.resize(letter, (128, 128), 0, 0, cv2.INTER_LINEAR)
	img = np.array(img, dtype = np.uint8)
	img = img.astype('float32')
	img = np.multiply(img, 1.0/255.0)
	x_batch.append(img.reshape(image_size, image_size))

x_batch = np.array(x_batch, dtype = np.uint8).reshape(n, image_size, image_size)
#print (x_batch.shape)

sess = tf.Session()
saver = tf.train.import_meta_graph('./model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

x = graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((n, 23)) 


testing = {x: x_batch, y_true: y_test_images}
result = sess.run(y_pred, feed_dict = testing)
print(result)