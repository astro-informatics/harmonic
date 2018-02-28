import numpy as np
cimport numpy as np
import chains as ch
import model as md
import evidence as cbe

# module to do 
# 1) sample spliting for training and using the data
# 2) cross validation on the models to chose hyper parameter


def split_data(chains not None, double training_proportion=0.5):    
    """Split the data in a chains instance into two (e.g. training and test
    sets) so that the new chains instances can be used for training and
    calculationg the evidence on the "test" set.

    Chains are split so that the first chains in the original chains object go 
    into the training set and the following go into the test set.

    Args:
        chains: Instance of a chains class containing the data to be split.
        training_proportion: The ratio of the data to be used in training 
            (default=0.5)

    Returns: (chains_train, chains_test)
        chains_train: Instance of a chains class containing chains to be used 
            to fit the model (e.g. training).
        chains_test: Instance of a chains class containing chains to be used
            to calculate the evidence (e.g. testing).

    Raises:
        ValueError: Raised if training_proportion is not strictly between 0
            and 1.
        ValueError: Raised if resulting nchains in training set is less than 1.
        ValueError: Raised if resulting nchains in test set is less than 1.
    """

    if training_proportion <= 0.0 or training_proportion >= 1.0:
        raise ValueError("training_proportion must be strictly between " \
            "0 and 1.")

    nchains_train = long(chains.nchains * training_proportion)
    nchains_test   = chains.nchains - nchains_train
    
    if nchains_train < 1:
        raise ValueError("nchains for training set must strictly greater " \
            "than 0.")
    if nchains_test < 1:
        raise ValueError("nchains for test set must strictly greater than 0.")

    ndim = chains.ndim

    chains_train = ch.Chains(ndim)
    chains_test   = ch.Chains(ndim)

    start_index = chains.start_indices[0]
    end_index   = chains.start_indices[nchains_train]
    chains_train.add_chains_2d_list(chains.samples[start_index:end_index,:],\
                                    chains.ln_posterior[start_index:end_index],\
                                    nchains_train, \
                                    chains.start_indices[:nchains_train+1])

    start_index = chains.start_indices[nchains_train]
    end_index   = chains.start_indices[-1]
    chains_test.add_chains_2d_list(chains.samples[start_index:end_index,:],\
                                   chains.ln_posterior[start_index:end_index],\
                                   nchains_test, \
                                   chains.start_indices[nchains_train:])

    return chains_train, chains_test

def validation_fit_indexes(long i_cross, long nchains_in_val_set, long ncross, list indexes):
    """ Function that pulls out the correct indexes for the chains of the
        validation and training sets

    Args:
        long i_cross: integer giving the cross validation iteration to perform
        long nchains_in_val_set: The number of chains that will go in each 
            validation set
        long ncross: The number of cross validation sets being made
        list indexes: T=A list with the suffled indexes

    Returns:
        list indexes_val: The list list of indexes for the validation set
        list indexes_fit: The list of indexes for the training set

    Raises:
        ValueError: If the value of i_cross doesn't fall between 0 and ncross-1
    """


    if i_cross < 0 or i_cross >= ncross:
        raise ValueError("i_cross is not the range set by ncross")

    cdef list indexes_val, indexes_fit

    if i_cross < ncross-1:
        indexes_val = indexes[i_cross*nchains_in_val_set:(i_cross+1)*nchains_in_val_set]
        indexes_fit = indexes[:i_cross*nchains_in_val_set] + indexes[(i_cross+1)*nchains_in_val_set:]
    else:
        indexes_val = indexes[(i_cross)*nchains_in_val_set:] # ensures all the chains get used even if nchains % ncross != 0
        indexes_fit = indexes[:i_cross*nchains_in_val_set]

    return indexes_val, indexes_fit


def cross_validation(chains, list domains, list hyper_parameters, \
                     long ncross=2, str MODEL="KernelDensityEstimate", \
                     long seed=-1, bint verbose=False):
    """ Splits data into ncross chunks. Then fits the model using
        each of the hyper parameters given using all but one of the 
        chunks. This procedure is done for all the chunks and the 
        average varience from all the chunks is used to decide which
        hyper parameters list was better.

    Args:
        chains: instance of a chains class with the data 
            trianed on
        list domains: The domains of the model's parameters
        list hyper_parameters: A list of length ncross where each entry
            is a hyper_parameters list to be trialed
        str MODEL: stirng identifying the model that is being cross 
            validated. Options are ("KernelDensityEstimate"),
            (default = "KernelDensityEstimate")
        long seed: seed for random number when drawing the chains,
            if this is negative the seed is not set
        bool verbose: Set to True to print results from cross validation
            evidence calculations (default=False)

    Returns:
        hyper_parameter list that was most succesful

        Raises:
            ValueError: If MODEL is not one of the posible models
    """

    cdef long i_cross, i_val, nchains_in_val_set
    cdef set posible_models
    cdef list indexes, indexes_val, indexes_fit, hyper_parameter

    cdef np.ndarray[double, ndim=2, mode='c'] validation_variences = np.zeros((ncross,len(hyper_parameters)))

    posible_models = {"HyperSphere", "KernelDensityEstimate"}

    if not MODEL in posible_models:
        raise ValueError("MODEL is not one of the possible models to cross validate")

    if seed > 0:
        np.random.seed(seed)

    indexes = list(np.random.permutation(chains.nchains))

    nchains_in_val_set = chains.nchains/ncross

    for i_cross in range(ncross):

        indexes_val, indexes_fit = validation_fit_indexes(i_cross, nchains_in_val_set, ncross, indexes)

        chains_val = chains.get_sub_chains(indexes_val)
        chains_fit = chains.get_sub_chains(indexes_fit)

        for i_val, hyper_parameter in enumerate(hyper_parameters):
            if MODEL == "HyperSphere":
                model = md.HyperSphere(chains.ndim, domains, hyper_parameters=hyper_parameter)
            if MODEL == "KernelDensityEstimate":
                model = md.KernelDensityEstimate(chains.ndim, domains, hyper_parameters=hyper_parameter)

            # foit model
            model.fit(chains_fit.samples,chains_fit.ln_posterior)

            # calculate evidence
            cal_ev = cbe.Evidence(chains_val.nchains, model)
            cal_ev.add_chains(chains_val)

            if verbose:
                print(MODEL, cal_ev.evidence_inv, cal_ev.evidence_inv_var, cal_ev.evidence_inv_var**0.5/cal_ev.evidence_inv, cal_ev.evidence_inv_var_var)

            validation_variences[i_cross,i_val] = cal_ev.evidence_inv_var

    return np.mean(validation_variences, axis=0)
