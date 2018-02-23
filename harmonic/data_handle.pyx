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