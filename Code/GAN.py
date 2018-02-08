
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('ggplot')
import pandas as pd 


# import xgboost as xgb

import pickle
import gc
import os
import sys

from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf

########################################### metrics ###########################################
def BaseMetrics(y_pred,y_true):
	TP = np.sum( (y_pred == 1) & (y_true == 1) )
	TN = np.sum( (y_pred == 0) & (y_true == 0) )
	FP = np.sum( (y_pred == 1) & (y_true == 0) )
	FN = np.sum( (y_pred == 0) & (y_true == 1) )
	return TP, TN, FP, FN

def SimpleMetrics(y_pred,y_true):
	TP, TN, FP, FN = BaseMetrics(y_pred,y_true)
	ACC = ( TP + TN ) / ( TP + TN + FP + FN )
	
	# Reporting
	from IPython.display import display
	print( 'Confusion Matrix')
	display( pd.DataFrame( [[TN,FP],[FN,TP]], columns=['Pred 0','Pred 1'], index=['True 0', 'True 1'] ) )
	print( 'Accuracy : {}'.format( ACC ))
	
def SimpleAccuracy(y_pred,y_true):
	TP, TN, FP, FN = BaseMetrics(y_pred,y_true)
	ACC = ( TP + TN ) / ( TP + TN + FP + FN )
	return ACC


def CheckAccuracy( x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2 ):
	import pandas as pd 
	from  sklearn import svm
	from sklearn.ensemble import RandomForestClassifier as rm
	from sklearn.ensemble import GradientBoostingClassifier as gbm

	dtrain = np.vstack( [ x[:int(len(x)/2)], g_z[:int(len(g_z)/2)] ] ) # Use half of each real and generated set for training
	dlabels = np.hstack( [ np.zeros(int(len(x)/2)), np.ones(int(len(g_z)/2)) ] ) # synthetic labels
	dtest = np.vstack( [ x[int(len(x)/2):], g_z[int(len(g_z)/2):] ] ) # Use the other half of each set for testing
	y_true = dlabels # Labels for test samples will be the same as the labels for training samples, assuming even batch sizes

	clf = gbm()
	clf = clf.fit(dtrain, dlabels)
	y_pred = clf.predict(dtest)
	return SimpleAccuracy(y_pred, y_true)
######################################### metrics ends #########################################

	
def get_data_batch(train, batch_size, seed=0):
	# # random sampling - some samples will have excessively low or high sampling, but easy to implement
	# np.random.seed(seed)
	
	# iterate through shuffled indices, so every sample gets covered evenly
	start_i = (batch_size * seed) % len(train)
	stop_i = start_i + batch_size
	shuffle_seed = (batch_size * seed) // len(train)
	np.random.seed(shuffle_seed)
	train_ix = np.random.choice( list(train.index), replace=False, size=len(train) ) # wasteful to shuffle every time
	train_ix = list(train_ix) + list(train_ix) # duplicate to cover ranges past the end of the set
	x = train.loc[ train_ix[ start_i: stop_i ] ].values
	
	return np.reshape(x, (batch_size, -1) )
	

	
def PlotData( x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, save=False, prefix='' ):
	
	real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
	gen_samples = pd.DataFrame(g_z, columns=data_cols+label_cols)
	
	f, axarr = plt.subplots(1, 2, figsize=(6,2) )
	if with_class:
		axarr[0].scatter( real_samples[data_cols[0]], real_samples[data_cols[1]], c=real_samples[label_cols[0]]/2 ) #, cmap='plasma'  )
		axarr[1].scatter( gen_samples[ data_cols[0]], gen_samples[ data_cols[1]], c=gen_samples[label_cols[0]]/2 ) #, cmap='plasma'  )
		
	else:
		axarr[0].scatter( real_samples[data_cols[0]], real_samples[data_cols[1]]) #, cmap='plasma'  )
		axarr[1].scatter( gen_samples[data_cols[0]], gen_samples[data_cols[1]]) #, cmap='plasma'  )
	axarr[0].set_title('real')
	axarr[1].set_title('generated')   
	axarr[0].set_ylabel(data_cols[1]) # Only add y label to left plot
	for a in axarr: a.set_xlabel(data_cols[0]) # Add x label to both plots
	axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim()) # Use axes ranges from real data for generated data
	
	if save:
		plt.save( prefix + '.xgb_check.png' )
		
	plt.show()

	

# #### Functions to define the layers of the networks used in the 'define_models' functions below
#function definiting generator network	
def generator_network(x, data_dim, base_n_count): 
	x = layers.Dense(base_n_count, activation='relu')(x)
	x = layers.Dense(base_n_count*2, activation='relu')(x)
	x = layers.Dense(base_n_count*4, activation='relu')(x)
	x = layers.Dense(data_dim)(x)    
	return x
	

#funtion defining discriminator network	
def discriminator_network(x, data_dim, base_n_count):
	x = layers.Dense(base_n_count*4, activation='relu')(x)
	# x = layers.Dropout(0.1)(x)
	x = layers.Dense(base_n_count*2, activation='relu')(x)
	# x = layers.Dropout(0.1)(x)
	x = layers.Dense(base_n_count, activation='relu')(x)
	x = layers.Dense(1, activation='sigmoid')(x)
	# x = layers.Dense(1)(x)
	return x
	
#function defining cretic network	
def critic_network(x, data_dim, base_n_count):
	x = layers.Dense(base_n_count*4, activation='relu')(x)
	x = layers.Dense(base_n_count*2, activation='relu')(x) # 2
	x = layers.Dense(base_n_count*1, activation='relu')(x) # 1
	x = layers.Dense(1)(x)
	return x

	
	
# #### Functions to define the keras network models    
	
def define_models_GAN(rand_dim, data_dim, base_n_count, type=None):
	generator_input_tensor = layers.Input(shape=(rand_dim, ))
	generated_image_tensor = generator_network(generator_input_tensor, data_dim, base_n_count)

	generated_or_real_image_tensor = layers.Input(shape=(data_dim,))
	
	if type == 'Wasserstein':
		discriminator_output = critic_network(generated_or_real_image_tensor, data_dim, base_n_count)
	else:
		discriminator_output = discriminator_network(generated_or_real_image_tensor, data_dim, base_n_count)

	generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')
	discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
									   outputs=[discriminator_output],
									   name='discriminator')

	combined_output = discriminator_model(generator_model(generator_input_tensor))
	combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')
	
	return generator_model, discriminator_model, combined_model

#### Functions specific to the vanilla GAN architecture   
#function for training GAN	
def training_steps_GAN(model_components):
	
	[ cache_prefix, with_class, starting_step,
						train, data_cols, data_dim,
						label_cols, label_dim,
						generator_model, discriminator_model, combined_model,
						rand_dim, nb_steps, batch_size, 
						k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
						data_dir, generator_model_path, discriminator_model_path, show,
						combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ] = model_components  
	
	for i in range(starting_step, starting_step+nb_steps):
		K.set_learning_phase(1) # 1 = train

		# train the discriminator
		for j in range(k_d):
			np.random.seed(i+j)
			z = np.random.normal(size=(batch_size, rand_dim))
			x = get_data_batch(train, batch_size, seed=i+j)
			
			if with_class:
				labels = x[:,-label_dim:]
				g_z = generator_model.predict([z, labels])
			else:
				g_z = generator_model.predict(z)
			
			d_l_r = discriminator_model.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
			d_l_g = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.001, size=batch_size)) # 0.0, 0.3 # GANs need noise to prevent loss going to zero
		disc_loss_real.append(d_l_r)
		disc_loss_generated.append(d_l_g)
		
		# train the generator
		for j in range(k_g):
			np.random.seed(i+j)
			z = np.random.normal(size=(batch_size, rand_dim))
			if with_class:
				# loss = combined_model.train_on_batch([z, labels], np.ones(batch_size)) # without noise
				loss = combined_model.train_on_batch([z, labels], np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
			else:
				# loss = combined_model.train_on_batch(z, np.ones(batch_size)) # without noise
				loss = combined_model.train_on_batch(z, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
		combined_loss.append(loss)
		
		# Determine xgb loss each step, after training generator and discriminator
		if not i % 10: # 2x faster than testing each step...
			K.set_learning_phase(0) # 0 = test
			test_size = 492 # test using all of the actual fraud data
			x = get_data_batch(train, test_size, seed=i)
			z = np.random.normal(size=(test_size, rand_dim))
			if with_class:
				labels = x[:,-label_dim:]
				g_z = generator_model.predict([z, labels])
				data_write = pd.DataFrame(g_z,columns = data_cols)
				data_write.to_csv('../Output/generated_data.csv',index = False)
			else:
				g_z = generator_model.predict(z)
				data_write = pd.DataFrame(g_z,columns = data_cols)
				data_write.to_csv('../Output/generated_data.csv',index = False)
				# data_write.to_csv('generated_data.csv',columns = data_cols,index = False)
			
			xgb_loss = CheckAccuracy( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim )
			xgb_losses = np.append(xgb_losses, xgb_loss)

		# Saving weights and plotting images
		if not i % log_interval:
			print('Step: {} of {}.'.format(i, starting_step + nb_steps))
			K.set_learning_phase(0) # 0 = test
						
			# loss summaries      
			print( 'Losses: G, D Gen, D Real, Xgb: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(combined_loss[-1], disc_loss_generated[-1], disc_loss_real[-1], xgb_losses[-1]) )
			print( 'D Real - D Gen: {:.4f}'.format(disc_loss_real[-1]-disc_loss_generated[-1]) )            
			print('Generator model loss: {}.'.format(combined_loss[-1]))
			print('Discriminator model loss gen: {}.'.format(disc_loss_generated[-1]))
			print('Discriminator model loss real: {}.'.format(disc_loss_real[-1]))
			print('svc loss: {}'.format(xgb_losses[-1]) )
			
			if show:
				print x.shape,'real data'
				print g_z.shape,'generated da'
				print data_cols, label_cols
				PlotData( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim, 
							save=False, prefix= data_dir + cache_prefix + '_' + str(i) )
			
	
	return [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]
	

#function to definine training complete network	
def adversarial_training_GAN(arguments, train, data_cols, label_cols=[], seed=0, starting_step=0):

	[rand_dim, nb_steps, batch_size, 
			 k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
			data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ] = arguments
	
	np.random.seed(seed)     # set random seed
	
	data_dim = len(data_cols)
	print('data_dim: ', data_dim)
	print('data_cols: ', data_cols)
	
	label_dim = 0
	with_class = False
	if len(label_cols) > 0: 
		with_class = True
		label_dim = len(label_cols)
		print('label_dim: ', label_dim)
		print('label_cols: ', label_cols)
	
	# define network models
	
	K.set_learning_phase(1) # 1 = train
	if with_class:
		cache_prefix = 'CGAN'
		generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count)
	else:
		cache_prefix = 'GAN'
		generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count)
	
	# compile models

	adam = optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.9)

	generator_model.compile(optimizer=adam, loss='binary_crossentropy')
	discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
	discriminator_model.trainable = False
	combined_model.compile(optimizer=adam, loss='binary_crossentropy')
	

	combined_loss, disc_loss_generated, disc_loss_real, xgb_losses = [], [], [], []
	
	if loss_pickle_path:
		print('Loading loss pickles')
		[combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
	if generator_model_path:
		print('Loading generator model')
		generator_model.load_weights(generator_model_path, by_name=True)
	if discriminator_model_path:
		print('Loading discriminator model')
		discriminator_model.load_weights(discriminator_model_path, by_name=True)

	model_components = [ cache_prefix, with_class, starting_step,
						train, data_cols, data_dim,
						label_cols, label_dim,
						generator_model, discriminator_model, combined_model,
						rand_dim, nb_steps, batch_size, 
						k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
						data_dir, generator_model_path, discriminator_model_path, show,
						combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ]
		
	[combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = training_steps_GAN(model_components)
		


		
def sample_z(m, n): # updated to normal distribution
#     return np.random.uniform(-1., 1., size=[m, n])
	return np.random.normal(size=[m, n])


def get_perturbed_batch(minibatch):
	return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)
