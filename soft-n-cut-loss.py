import cv2
import numpy as np 
import tensorflow as tf 
import numpy as np
def edge_weights(flatten_image, rows , cols, std_intensity=5, std_position=5, radius=9):
	'''
	Inputs :
	flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels) 
	std_intensity : standard deviation for intensity 
	std_position : standard devistion for position
	radius : the length of the around the pixel where the weights 
	is non-zero
	rows : rows of the original image (unflattened image)
	cols : cols of the original image (unflattened image)

	Output : 
	weights :  2d tf array edge weights in the pixel graph

	Used parameters :
	n : number of pixels 
	'''
	n = rows*cols
	w = tf.zeros([n,n])
	A = outer_product(flatten_image, tf.ones_like(flatten_image))
	A_T = tf.transpose(A)
	ele_diff = tf.exp(-1*tf.square((tf.divide((A - A_T), std_intensity))))

	xx, yy = tf.meshgrid(tf.range(rows), tf.range(cols))
	distance_matrix = tf.exp(-1*tf.divide(tf.square(xx - yy), std_position**2))
	# ele_diff = tf.reshape(ele_diff, (rows, cols))
	# w = ele_diff + distance_matrix
	'''
	for i in range(n):
		for j in range(n):
			# because a (x,y) in the original image responds in (x-1)*cols + (y+1) in the flatten image
			x_i= (i//cols) +1 
			y_i= (i%cols) - 1
			x_j= (j//cols) + 1
			y_j= (j%cols) - 1
			distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
			if (distance < radius):
				w[i][j] = tf.exp(-((flatten_image[i]- flatten_image[j])/std_intensity)**2) * tf.exp(-(distance/std_position)**2)
	# return w as a lookup table			
	'''
	return w

def outer_product(v1,v2):
	'''
	Inputs:
	v1 : m*1 tf array
	v2 : 1*m tf array

	Output :
	v1 x v2 : m*m array
	'''
	v1 = tf.expand_dims((v1), axis=0)
	v2 = tf.expand_dims((v2), axis=0)
	print(v2.get_shape())
	return tf.matmul(tf.transpose(v1),(v2))


def numerator(k_class_prob,weights):
	'''
	Inputs :
	k_class_prob : k_class pixelwise probability (rows*cols) tensor 
	weights : edge weights n*n tensor 
	'''
	return tf.reduce_sum(tf.multiply(weights,outer_product(k_class_prob,k_class_prob)))

def denominator(k_class_prob,weights):	
	'''
	Inputs:
	k_class_prob : k_class pixelwise probability (rows*cols) tensor
	weights : edge weights	n*n tensor 
	'''

	return tf.reduce_sum(tf.multiply(weights,outer_product(k_class_prob,tf.ones(tf.shape(k_class_prob)))))

def soft_n_cut_loss(flatten_image,prob, k, rows, cols):
	'''
	Inputs: 
	prob : (rows*cols*k) tensor 
	k : number of classes (integer)
	flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
	rows : number of the rows in the original image
	cols : number of the cols in the original image

	Output : 
	soft_n_cut_loss tensor for a single image

	'''

	soft_n_cut_loss = tf.Variable(k)
	weights = edge_weights(flatten_image, rows ,cols)
	# for t in range(k): 
	# 	soft_n_cut_loss = soft_n_cut_loss - (numerator(prob[:,:,t],weights)/denominator(prob[:,:,t],weights))

	return weights
	# return soft_n_cut_loss


image = tf.ones([256*256], dtype=tf.float64)
prob = tf.ones([256, 256, 3])
loss = soft_n_cut_loss(image, prob, 3, 256,256)
