import tensorflow as tf
#tf.enable_eager_execution()

from tensorflow.python.keras.applications.vgg19 import preprocess_input, VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import time

result_path = "result.jpg"
content_img_path = "content.jpg"
style_img_path = "style.jpg"


######################################
#          Hyperparameters           #

content_layer = 'block5_conv2'

style_layers =[

		'block1_conv1',
		'block2_conv1',
		'block3_conv1',
		'block5_conv1' 
		]

iterations=20
alpha = 10.
beta = 20.
learning_rate = 7.

#######################################

def load_and_preprocess_img(image_path):
	img = load_img(image_path)
	img = img_to_array(img)
	img = preprocess_input(img)
	img = np.expand_dims(img,axis=0)

	return img

#To help during visualization. Doing the reverse of preprocess
def deprocess(x):
	x[:,:,0]+=103.939
	x[:,:,1]+=116.779
	x[:,:,2]+=123.68
	x=x[:,:,::-1]
	x = np.clip(x,0,255).astype('uint8')

	return x

def display_image(image):
	if len(image.shape) == 4:
		img = np.squeeze(image,axis=0)
	img=deprocess(img)
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img)
	plt.show()
	return 

def content_cost(content,generated):
	a_C = content_model(content)
	a_G = content_model(generated)
	cost = tf.reduce_mean(tf.square(a_C - a_G))
	return cost
#Gram matric is used to match feature distribution

def gram_matrix(A):
	n_C = int(A.shape[-1])
	a = tf.reshape(A,[-1,n_C])
	print(a.shape)

	n = tf.shape(a)[0]
	G = tf.matmul(a,a,transpose_a=True)

	return G / tf.cast(n,tf.float32)



def style_cost(style,generated):
	lan = 1. / len(style_models)
	J_style = 0

	for style_model in style_models:
		a_S = style_model(style)
		a_G = style_model(generated)
		GS = gram_matrix(a_S)
		GG = gram_matrix(a_G)

		current_cost = tf.reduce_mean(tf.square(GS - GG))
		J_style += current_cost * lan

	return J_style

def training_loop(content_path,style_path,iterations=iterations,
		alpha=alpha,beta=beta,learning_rate=learning_rate):
	generated_imgs=[]
	content = load_and_preprocess_img(content_path)
	style = load_and_preprocess_img(style_path)

	generated = tf.contrib.eager.Variable(content,dtype=tf.float32)

	opt = tf.train.AdamOptimizer(learning_rate = learning_rate)

	best_cost = 1e12 + 0.1
	best_image = None

	start_time = time.time()

	for i in range(iterations):
		with tf.GradientTape() as tape:
			J_content = content_cost(content,generated)
			J_style = style_cost(style,generated)
			J_total = (alpha * J_content )+ (beta * J_style)

		grads = tape.gradient(J_total,generated)
		opt.apply_gradients([(grads,generated)])

		if J_total < best_cost:
			best_cost = J_total
			best_image = generated.numpy()

		print('Cost at {}: {}. Time elapsed: {}'.format(i,J_total,time.time()-start_time))

		generated_images.append(generated)

	return best_image, generated_images



model = VGG19(include_top = False,
		weights='imagenet')
model.trainable= False
model.summary()

display_image(load_and_preprocess_img(style_img_path))
display_image(load_and_preprocess_img(content_img_path))


content_model = Model(model.input,model.get_layer(content_layer).output)

style_models = [Model(model.input,model.get_layer(layer).output) for layer in style_layers]

best_image, generated_images= training_loop(content_img_path,style_img_path)

display_image(best_image)

plt.figure(figsize=(30,30))

#Displays other generated images during training.
for i,j in enumerate(generated_images):
	plt.subplots(5,4,i+1)
	display_image(j)

cv2.imwrite(result_path,best_image)
