"""

Code for generating synthetic data to overcome imbalance data problem 

# -*- coding: utf-8 -*-

#POC Code  :  Synthetic Data Generation Using Generative Adversial Network :  H2H DATA
Synthetic Data Generation.
Models: GAN model is used


#Copyright@ H2H DATA

#The entire prcess occurs in seven stages-
# 1. DATA INGESTION
# 2. DATA ANALYSIS 
# 3. DATA MUNGING
# 4. DATA EXPLORATION
# 5. DATA MODELING
# 6. HYPER-PARAMETERS OPTIMIZATION
# 7. PREDICTION
# 8. VISUAL ANALYSIS
# 9. RESULTS


Used library
1. pandas
2. numpy
3. time
4. sklearn
5. matplotlib
6. GAN
7. Keras
"""




import GAN
import importlib
from GAN import *
import matplotlib.pyplot as plt

def GAN_model():


	########################################## data ingestion ###############################
	'''
	Data Description:
	target:'Class' value:0/1

	features:
	Name: 'Time',   value: real
	Name: 'V1',		value: real
	Name: 'V2', 	value: real
	Name: 'V3', 	value: real
	Name: 'V4',		value: real
	Name: 'V5', 	value: real
	Name: 'V6',		value: real
	Name: 'V7', 	value: real
	Name: 'V8', 	value: real
	Name: 'V9',		value: real
	Name: 'V10',	value: real
    Name: 'V11', 	value: real
    Name: 'V12', 	value: real
    Name: 'V13', 	value: real
    Name: 'V14', 	value: real
    Name: 'V15',	value: real
    Name: 'V16', 	value: real
    Name: 'V17',	value: real
    Name: 'V18', 	value: real
    Name: 'V19', 	value: real
    Name: 'V20',	value: real
    Name: 'V21', 	value: real
    Name: 'V22', 	value: real
    Name: 'V23', 	value: real
    Name: 'V24',	value: real
    Name: 'V25',	value: real
    Name: 'V26',	value: real
    Name: 'V27',	value: real
    Name: 'V28', 	value: real
    Name: 'Amount', value: real
       
	'''
	import pandas as pd 
	train = pd.read_csv('../Data/creditcard.csv')
	train_start = train
	####################################### data ingestion  ends ############################

	####################################### data Analysis ###################################
	'''
	Analysing data
	'''
	plt.figure()
	train.diff().hist(color='k', alpha=0.5, bins=50)
	plt.show()
	train.plot.box()
	plt.show()
	####################################### data Analysis ends ##############################




	####################################### data exploration ###################################
	'''
	Making data ready to for model
	1. taking log of amount for proper distribution
	2. centering and scaling data using middle 99.8%
	3. seperating label from features 
	'''
	d_log = np.log10( train['Amount'].values + 1 )
	train['Amount'] = d_log


	data_cols = list(train.columns[ train.columns != 'Class' ])
	percentiles =  pd.DataFrame( np.array([ np.percentile( train[i], [ 0.1, 99.9 ] ) for i in data_cols ]).T,
							columns=data_cols, index=['min','max'] )

	percentile_means = \
		[ [ np.mean( train.loc[ (train[i]>percentiles[i]['min']) & (train[i]<percentiles[i]['max']) , i ] ) ]
		 for i in data_cols ]

	percentiles = percentiles.append( pd.DataFrame(np.array(percentile_means).T, columns=data_cols, index=['mean']) )

	percentile_stds = \
		[ [ np.std( train.loc[ (train[i]>percentiles[i]['min']) & (train[i]<percentiles[i]['max']) , i ] ) ]
		 for i in data_cols ]

	percentiles = percentiles.append( pd.DataFrame(np.array(percentile_stds).T, columns=data_cols, index=['stdev']) )
	train[data_cols] = ( train[data_cols] - percentiles.loc[ 'mean', data_cols ] ) / percentiles.loc[ 'stdev', data_cols ]

	train_model= train
	train = train.loc[ train['Class']==1 ].copy()

	label_cols = [ i for i in train.columns if 'Class' in i ]
	data_cols = [ i for i in train.columns if i not in label_cols ]
	train[ data_cols ] = train[ data_cols ] / 10 # scale to random noise size, one less thing to learn
	train_no_label = train[data_cols]
	########################################## data exploration ends #############################




	########################################## modeling ##########################################
	'''
	training GAN network and generating synthetic data
	'''

	data_dim = 32 # 32 # needs to be ~data_dim
	base_n_count = 128 # 128
	steps = 500 + 1 # 50000 # Add one for logging of the last interval
	batch_size = 128 # 64
	k_d = 1  # number of critic network updates per adversarial training step
	k_g = 1  # number of generator network updates per adversarial training step
	critic_pre_train_steps = 100 # 100  # number of steps to pre-train the critic before starting 
											#adversarial training
	log_interval = 100 # 100  # interval (in steps) at which to log loss summaries and save plots of 
										#image samples to disc
	learning_rate = 5e-4 # 5e-5
	data_dir = 'cache/'
	generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None
	show = True 

	# Training the vanilla GAN 
	k_d = 1  # number of critic network updates per adversarial training step
	learning_rate = 5e-4 # 5e-5
	arguments = [data_dim, steps, batch_size, 
				 k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
				data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ]

	adversarial_training_GAN(arguments, train_no_label, data_cols ) # GAN
	########################################## modeling ends ######################################


	########################################## prediction  #########################################
	'''
	take generated data and combines it with existing data and checkes f1 score using mlp 
	'''
	generated_data = pd.read_csv('generated_data.csv')
	generated_data['Class'] = [0]*len(generated_data)
	generated_data.rename(columns=dict(zip(generated_data.columns, train_model.columns)), inplace=True)
	data = train_model.append(generated_data)

	from sklearn.utils import shuffle
	from sklearn.neural_network import MLPClassifier
	from sklearn.model_selection import cross_val_score
	data = shuffle(data)
	clf_mlp = MLPClassifier(hidden_layer_sizes = (10,4) ,warm_start = True)
	target = data['Class'].values
	features = data.drop(['Class'],axis = 1).values
	print  cross_val_score(clf_mlp,features,target,scoring = 'f1'), 'f1 score using mlp after data generation'
	########################################## prediction ends #####################################




if __name__ == '__main__':
	GAN_model()
