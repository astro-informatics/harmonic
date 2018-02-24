import numpy as np
cimport numpy as np
import chains as ch

# module to do 
# 1) sample spliting for training and using the data
# 2) cross validation on the models to chose hyper parameter


def split_data(chains not None, split_ratio=0.5):
	"""splits the data in a chains instance into two
	   so the new chains instances can be used for testing
	   and calculationg the evidence.

	Args:
		chains: instance of a chains class with the data 
			to be split
		split_ratio: The ratio of the data to be used in
			training (default=0.5)

	Returns:
		chains_train: instance of a chains class to be used
			to fit the model
		chains_use: instance of a chains class to be used
			to calculate the evidence



	"""

	nchains_train = int(chains.nchains * split_ratio)
	nchains_use   = chains.nchains - nchains_train

	ndim = chains.ndim

	chains_train = ch.Chains(ndim)
	chains_use   = ch.Chains(ndim)

	start_index = chains.start_indices[0]
	end_index   = chains.start_indices[nchains_train]
	chains_train.add_chains_2d_list(chains.samples[start_index:end_index,:],\
									chains.ln_posterior[start_index:end_index],\
									nchains_train, \
									chains.start_indices[:nchains_train+1])

	start_index = chains.start_indices[nchains_train]
	end_index   = chains.start_indices[-1]
	chains_use.add_chains_2d_list(chains.samples[start_index:end_index,:],\
									chains.ln_posterior[start_index:end_index],\
									nchains_use, \
									chains.start_indices[nchains_train:])

	return chains_train, chains_use


def cross_validation(chains, domains, hyper_parameters, ncross=2, MODEL="KernalDensityEstimation"):
	""" Splits data into ncross chunks. Then fits the model using
		each of the hyper parameters given using all but one of the 
		chunks. This procedure is done for all the chunks and the 
		average varience from all the chunks is used to decide which
		hyper parameters list was better.

	Args:
		chains: instance of a chains class with the data 
			trianed on
		domains: The domains of the model's parameters
		hyper_parameters: A list of length ncross where each entry
			is a hyper_parameters list to be trialed

	Returns:
		hyper_parameter list that was most succesful

		Raises:
	"""

	if MODEL == "KernalDensityEstimation":
		print("using ", MODEL)
