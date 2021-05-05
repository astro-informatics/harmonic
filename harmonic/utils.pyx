import numpy as np
cimport numpy as np
import chains as ch
import model as md
import evidence as cbe
import logs as lg 


def split_data(chains not None, double training_proportion=0.5):    
    """Split the data in a chains instance into two (e.g. training and test sets).

    New chains instances can be used for training and calculation the evidence
    on the "test" set.

    Chains are split so that the first chains in the original chains object go 
    into the training set and the following go into the test set.

    Args:

        chains (Chains): Instance of a chains class containing the data to be split.

        training_proportion (double): Proportion of data to be used in training
            (default=0.5)

    Returns:

        (Chains, Chains): A tuple containing the following two Chains.

            - chains_train (Chains): Instance of a chains class containing
              chains to be used to fit the model (e.g. training).

            - chains_test (Chains): Instance of a chains class containing
              chains to be used to calculate the evidence (e.g. testing).

    Raises:

        ValueError: Raised if training_proportion is not strictly between 0 and
            1.

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


def validation_fit_indexes(long i_fold, long nchains_in_val_set, long nfold,
                           list indexes):
    """Pull out the correct indexes for the chains of the validation and training 
    sets.

    Args:

        i_fold (long): Cross validation iteration to perform.

        nchains_in_val_set (long): The number of chains that will go in each
            validation set. 

        nfold (long): Number of fold validation sets to be made.

        indexes (list): List of the chains to be used in fold validation that
            need to be split.

    Returns:

        (list, list): A tuple containing the following two lists of indices.

            - indexes_val (list): List of indexes for the validation set.

            - indexes_fit (list): List of indexes for the training set.

    Raises:

        ValueError: Raised if the value of i_fold does not fall between 0 and
            nfold-1.

    """

    if i_fold < 0 or i_fold >= nfold:
        raise ValueError("i_fold is not the range set by nfold")

    if nchains_in_val_set < 1 or nchains_in_val_set >= len(indexes):
        raise ValueError("nchains_in_val_set must be strictly between 0 " \
            "and length of indexes.")

    cdef list indexes_val, indexes_fit

    if i_fold < nfold-1:
        indexes_val = indexes[i_fold*nchains_in_val_set: \
                              (i_fold+1)*nchains_in_val_set]
        indexes_fit = indexes[:i_fold*nchains_in_val_set] \
            + indexes[(i_fold+1)*nchains_in_val_set:]
    else:
        indexes_val = indexes[(i_fold)*nchains_in_val_set:] 
        # ensures all the chains get used even if nchains % nfold != 0
        indexes_fit = indexes[:i_fold*nchains_in_val_set]

    return indexes_val, indexes_fit


def cross_validation(chains, 
                     list domains, 
                     list hyper_parameters, 
                     long nfold=2, 
                     modelClass = md.KernelDensityEstimate, 
                     long seed=-1):    
    """Perform n-fold validation for given model using chains to be split into
    validation and training data.
    
    First, splits data into nfold chunks. Second, fits the model using each of 
    the hyper parameters given using all but one of the chunks (the validation 
    chunk). This procedure is performed for all the chunks and the average 
    (mean) log-space variance from all the chunks is computed and returned.  
    This can be used to decide which hyper parameters list was better.

    Args:

        chains (Chains): Chains containing samples (to be split into
            training and validation data herein).

        domains (list): Domains of the model's parameters.

        hyper_parameters (list): List of hyper_parameters where each entry is a
            hyper_parameter list to be considered.

        modelClass (Model): Model that is being cross validated (default =
            KernelDensityEstimate).

        seed (long): Seed for random number generator when drawing the chains
            (if this is negative the seed is not set).

    Returns:

        (list): Mean log validation variance (averaged over nfolds) for each hyperparameter.

    Raises:

        ValueError: Raised if model is not one of the posible models.

    """

    cdef long i_fold, i_val, nchains_in_val_set
    cdef set posible_models
    cdef list indexes, indexes_val, indexes_fit, hyper_parameter

    cdef np.ndarray[double, ndim=2, mode='c'] ln_validation_variances = \
            np.zeros((nfold,len(hyper_parameters)))

    if seed > 0:
        np.random.seed(seed)

    indexes = list(np.random.permutation(chains.nchains))

    nchains_in_val_set = chains.nchains/nfold

    for i_fold in range(nfold):

        indexes_val, indexes_fit = validation_fit_indexes(i_fold, 
                                                          nchains_in_val_set,
                                                          nfold, indexes)

        chains_val = chains.get_sub_chains(indexes_val)
        chains_fit = chains.get_sub_chains(indexes_fit)

        for i_val, hyper_parameter in enumerate(hyper_parameters):
            
            model = modelClass(chains.ndim, domains, 
                               hyper_parameters=hyper_parameter)

            # Fit model
            model.fit(chains_fit.samples,chains_fit.ln_posterior)

            # Calculate evidence
            ev = cbe.Evidence(chains_val.nchains, model)
            ev.add_chains(chains_val)

            lg.debug_log('cross_validation: ifold = {}; hyper_parameter = {}'
                  .format(i_fold, hyper_parameter))
            lg.debug_log('cross_validation: evidence_inv = {}'
                  .format(ev.evidence_inv))
            lg.debug_log('cross_validation: evidence_inv_var = {}'
                  .format(ev.evidence_inv_var))
            lg.debug_log('cross_validation:' + 
                  ' evidence_inv_var**0.5/evidence_inv = {}'
                  .format(ev.evidence_inv_var**0.5/ev.evidence_inv))
            lg.debug_log('cross_validation: evidence_inv_var_var = {}'
                  .format(ev.evidence_inv_var_var))

            ln_validation_variances[i_fold,i_val] = ev.ln_evidence_inv_var

    return np.nanmean(ln_validation_variances, axis=0)
