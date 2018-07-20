import cv2
import numpy as np 
import tensorflow as tf 
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Conv2D, Conv3D,
                                            Dense, Flatten,
                                            GlobalAveragePooling2D,
                                            GlobalMaxPooling2D, Input,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, Dropout, concatenate,
											UpSampling2D)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K_B
import coloredlogs
from os.path import exists
from input_data import input_data


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
	A = outer_product(flatten_image, tf.ones_like(flatten_image))
	A_T = tf.transpose(A)
	intensity_weight = tf.exp(-1*tf.square((tf.divide((A - A_T), std_intensity))))

	xx, yy = tf.meshgrid(tf.range(rows), tf.range(cols))
	xx = tf.reshape(xx, (rows*cols,))
	yy = tf.reshape(yy, (rows*cols,))
	A_x = outer_product(xx, tf.ones_like(xx))
	A_y = outer_product(yy, tf.ones_like(yy))

	xi_xj = A_x - tf.transpose(A_x)
	yi_yj = A_y - tf.transpose(A_y)

	sq_distance_matrix = tf.square(xi_xj) + tf.square(yi_yj)

	dist_weight = tf.exp(-tf.divide(sq_distance_matrix,tf.square(std_position)))
	dist_weight = tf.cast(dist_weight, tf.float32)
	print (dist_weight.get_shape())
	print (intensity_weight.get_shape())
	weight = tf.multiply(intensity_weight, dist_weight)


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
	return weight

def outer_product(v1,v2):
	'''
	Inputs:
	v1 : m*1 tf array
	v2 : m*1 tf array

	Output :
	v1 x v2 : m*m array
	'''
	v1 = tf.reshape(v1, (-1,))
	v2 = tf.reshape(v2, (-1,))
	v1 = tf.expand_dims((v1), axis=0)
	v2 = tf.expand_dims((v2), axis=0)
	return tf.matmul(tf.transpose(v1),(v2))

def numerator(k_class_prob,weights):

	'''
	Inputs :
	k_class_prob : k_class pixelwise probability (rows*cols) tensor 
	weights : edge weights n*n tensor 
	'''
	k_class_prob = tf.reshape(k_class_prob, (-1,))	
	return tf.reduce_sum(tf.multiply(weights,outer_product(k_class_prob,k_class_prob)))

def denominator(k_class_prob,weights):	
	'''
	Inputs:
	k_class_prob : k_class pixelwise probability (rows*cols) tensor
	weights : edge weights	n*n tensor 
	'''
	k_class_prob = tf.cast(k_class_prob, tf.float32)
	k_class_prob = tf.reshape(k_class_prob, (-1,))	
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

	soft_n_cut_loss = k
	weights = edge_weights(flatten_image, rows ,cols)
	
	for t in range(k): 
		soft_n_cut_loss = soft_n_cut_loss - (numerator(prob[:,:,t],weights)/denominator(prob[:,:,t],weights))

	return soft_n_cut_loss
	# return soft_n_cut_loss

if __name__ == '__main__':
	'''
	image = tf.ones([224*224])
	prob = tf.ones([224, 224,2])/2
	loss = soft_n_cut_loss(image, prob, 2, 224, 224)

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		print(sess.run(loss))
		# print (sess.run(w))
 	'''
	img_rows = 64
	img_cols = 64
	num_classes = 2
	bn_axis=3
	display_step = 10
	logdir = "checkpoints/logs"
	checkpt_dir_ckpt = "checkpoints/trained.ckpt"
	checkpt_dir = "checkpoints"

	x = tf.placeholder(tf.float32, shape=[None, img_rows, img_cols, 3], name="input")
	global_step_tensor = tf.train.get_or_create_global_step()
	
	def enc_conv_block(inputs, filters=[128,128], kernel_size=[3,3], activation='relu', kernel_initializer='he_normal', block='', module='', pre_pool=True):
		fa, fb = filters
		ka, kb = kernel_size
		conv1 = Conv2D(fa, ka, activation=activation, padding='same', kernel_initializer=kernel_initializer, name=module+'_conv_enc_'+block+'_1')(inputs)
		conv1 = Conv2D(fb, kb, activation=activation, padding='same', kernel_initializer=kernel_initializer, name=module+'_conv_enc_'+block+'_2')(conv1)
		conv1 = BatchNormalization(axis=bn_axis, name=module+'_bn_enc_'+block+'_3')(conv1)
		pool1 = MaxPooling2D(pool_size=(2,2), name=module+'_maxpool_enc_'+block+'_4')(conv1)
		tf.summary.histogram(module+'_maxpool_enc_'+block+'_4',pool1)
		if not pre_pool:
			return pool1
		else:
			return conv1,pool1

	def dec_conv_block(inputs, filters=[128, 128, 128], kernel_size=[2,3,3], activation='relu', kernel_initializer='he_normal', block='', module=''):
		previous_layer, concat_layer = inputs
		fa, fb, fc = filters
		ka, kb, kc = kernel_size
		up1 = Conv2D(fa, ka, activation=activation, padding='same', kernel_initializer=kernel_initializer, name=module+'_conv_dec_'+block+'_2')(UpSampling2D(size=(2,2), name=module+'_upsam_block_'+block+'_1')(previous_layer))
		# print (up1.get_shape())
		merge1 = concatenate([concat_layer, up1], name=module+'_concat_'+block+'_3')
		conv2 = Conv2D(fb, kb, activation=activation, padding='same', kernel_initializer=kernel_initializer, name=module+'_conv_dec_'+block+'_4')(merge1)
		conv3 = Conv2D(fc, kc, activation=activation, padding='same', kernel_initializer=kernel_initializer,name=module+'_conv_dec_'+block+'_5')(conv2)
		conv3 = BatchNormalization(axis=bn_axis, name=module+'_bn_dec_'+block+'_6')(conv3)
		tf.summary.histogram(module+'_bn_dec_'+block+'_6', conv3)
		return conv3

	def join_enc_dec(inputs, filters=[1024,1024], kernel=[3,3],activation='relu', kernel_initializer='he_normal', module='', block='join'):	
		fa, fb = filters
		ka, kb = kernel
		conv1 = Conv2D(fa, ka, activation=activation, padding='same', kernel_initializer=kernel_initializer, name=module+"_join_conv_1")(inputs)
		conv1 = Conv2D(fb, kb, activation=activation, padding='same', kernel_initializer=kernel_initializer, name=module+"_join_conv_2")(conv1)
		conv1 = BatchNormalization(axis=bn_axis, name=module+'_join_bn_3_')(conv1)
		conv1 = Dropout(0.5, name=module+'_join_dropout_4')(conv1)
		tf.summary.histogram(module+'_join_bn_3_', conv1)
		return conv1
	
	def unet(input_size=(-1,img_rows,img_cols,3), input_tensor=None, output_layers=1,module=''):
		
		if input_tensor is None:
			inputs = Input(input_size)
		else:
			inputs = input_tensor
		bn_axis=3
		prepool_1, layer1 = enc_conv_block(inputs, [64, 64], [3,3], block='a', module=module)
		prepool_2, layer2 = enc_conv_block(layer1, [128,128], [3,3], block='b', module=module)
		prepool_3, layer3 = enc_conv_block(layer2, [256,256], [3,3], block='c', module=module)
		prepool_4, layer4 = enc_conv_block(layer3, [512,512], [3,3], block='d', module=module)

		layer4 = Dropout(0.5)(layer4)

		join_layer = join_enc_dec(layer4, [1024,1024], [3,3], module=module)
		
		layer4 = dec_conv_block([join_layer, prepool_4], [512,512,512], [2,3,3], block='d', module=module)
		layer3 = dec_conv_block([layer4, prepool_3], [256,256,256], [2,3,3], block='c', module=module)
		layer2 = dec_conv_block([layer3, prepool_2], [128,128,128], [2,3,3], block='b', module=module)
		layer1 = dec_conv_block([layer2, prepool_1], [64,64,64], [2,3,3], block='a', module=module)

		output = Conv2D(output_layers, 1, kernel_initializer='he_normal', name=module+'_output_layer')(layer1)

		return output
	

	'''
	def unet(pretrained_weights = None,input_size = (-1,img_rows,img_cols,3),  input_tensor = None, output_layers = 1, module='O', name='O'):
		if input_tensor is None:
			inputs = Input(input_size)
		else:
			inputs = input_tensor
		bn_axis = 3
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		conv1 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv1')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		conv2 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv2')(conv2)		
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		conv3 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv3')(conv3)		
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		conv4 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv4')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		conv5 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv5')(conv5)
		drop5 = Dropout(0.5)(conv5)
		print (drop5.get_shape())
		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		print (up6.get_shape())
		print (drop4.get_shape())
		merge6 = concatenate([drop4,up6], axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
		conv6 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv6')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = concatenate([conv3,up7], axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		conv7 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv7')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2,up8], axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		conv8 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv8')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv9')(conv9)
		
		conv10 = Conv2D(output_layers, 1)(conv9)
		conv10 = BatchNormalization(axis=bn_axis, name=name+'_bn_conv10')(conv10)

		# model = Model(input = inputs, output = conv10)

		# model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		
		#model.summary()

		# if(pretrained_weights):
		# 	model.load_weights(pretrained_weights)

		return conv10
	'''
	def encoder(num_classes, input_shape=[-1,img_rows,img_cols,3], input_tensor = None):
		if input_tensor is None:
			img_input = Input(shape=input_shape)
		else:
			img_input = input_tensor
		x = unet(input_tensor = img_input, output_layers=num_classes, module='ENCODER')
		x = tf.nn.softmax(x, axis=2)
		return (x)
	def decoder(input_shape=[-1, img_rows,img_cols,3], input_tensor=None):
		if input_tensor is None:
			img_input = Input(shape=input_shape)
		else:
			img_input = input_tensor
		x = unet(input_tensor = img_input, output_layers=3, module='DECODER') # 3 because  of number of channels
		return (x)
	
	coloredlogs.install(level='DEBUG')
	tf.logging.set_verbosity(tf.logging.DEBUG)
	
	output = encoder(num_classes, input_tensor = x)
	decode = decoder(input_tensor=output)
	x_yuv = tf.image.rgb_to_yuv(x)
	loss = soft_n_cut_loss(tf.reshape(x_yuv[:,:,:,0], (img_cols*img_rows,)), tf.reshape(output, (img_rows, img_cols, num_classes)), num_classes, img_rows, img_cols)
	recons_loss = tf.reduce_mean(tf.square(x - decode))
	
	vars_encoder = [var for var in tf.trainable_variables() if var.name.startswith("ENCODER")]
	vars_trainable = [var for var in tf.trainable_variables()]

	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
	op_recons = optimizer.minimize(recons_loss, global_step = global_step_tensor, var_list=vars_trainable)
	op = optimizer.minimize(loss, global_step=global_step_tensor, var_list=vars_encoder)
	
	tf.summary.scalar('soft_n_cut_loss', loss)
	tf.summary.scalar('reconstruction_loss', recons_loss)
	tf.summary.image('output_image', decode)
	tf.summary.image('input_image', x)
	tf.summary.image('segmented_op', tf.reshape(output[:,:,:,0], (1, img_rows, img_cols, 1)))
	tf.summary.histogram('segmented_image', output)
	tf.summary.histogram('reconstructed_image', decode)

	merged = tf.summary.merge_all()
	saver = tf.train.Saver()
	
	with K_B.get_session() as sess:
		train_writer = tf.summary.FileWriter(logdir,sess.graph)
		
		init = tf.global_variables_initializer()
		sess.run(init)
		
		if exists(checkpt_dir):
			if tf.train.latest_checkpoint(checkpt_dir) is not None:
				tf.logging.info('Loading Checkpoint from '+ tf.train.latest_checkpoint(checkpt_dir))
				saver.restore(sess, tf.train.latest_checkpoint(checkpt_dir))
		else:
			tf.logging.info('Training from Scratch -  No Checkpoint found')
		
		iterator = input_data()
		next_items = iterator.get_next()

		# img_lab = np.expan/d_dims(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), axis=0)
		i = 0
		while True:
			batch_x = sess.run(next_items)
			# print (batch_x)
			_ = sess.run([op], feed_dict={x:batch_x})
			gst, _=  sess.run([global_step_tensor, op_recons], feed_dict={x:batch_x})
			i+=1
			if i%display_step ==0:
				soft_loss, reconstruction_loss, summary, segment, output_image =  sess.run([loss, recons_loss, merged, output, decode], feed_dict={x:batch_x})
				train_writer.add_summary(summary, gst)
				tf.logging.info("Iteration: " + str(gst) + " Soft N-Cut Loss: " + str(soft_loss) + " Reconstruction Loss " + str(reconstruction_loss))
				saver.save(sess, checkpt_dir_ckpt, global_step=tf.train.get_global_step())