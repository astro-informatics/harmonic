import numpy as np
import bilby
import emcee
import matplotlib.pyplot as plt
import harmonic as hm
import os
import pymultinest
import getdist
from getdist import plots
getdist.chains.print_load_details = False


""" 
Example of 4D parameter estimation of Binary Black Hole Merger simulated data
using Bilby software. 

Comparing results to Pymultinest as analytical evidence is intractible at this
higher dimension.

"""

class gw_likelihood_based():

    """Note: each method doesn't actually return enything, it just adds it to 
    self so it can be caleld once the method has been run!
    """

    def __init__(self, seed=None):

        """Set seed=None to keep it stochastic
        Note - for sbi simulator add search parameters as a parameter rather than a method

        """
        
        np.random.seed(seed)

        self.duration = 4.
        self.sampling_frequency = 2048.
        self.minimum_frequency = 20
        self.reference_frequency = 50.
        self.ifos_keys = ['H1', 'L1', 'V1']

        self.waveform_model = 'IMRPhenomPv2'

        self.search_parameter_keys = ["mass_1", "mass_2", "luminosity_distance", "theta_jn"]
        self.fixed_parameter_keys = ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'phase', 'geocent_time', 'ra', 'dec']

        self.search_parameters = [36., 29., 2000., 0.4]
        self.fixed_parameters = [0.4, 0.3, 0.5, 1.0, 1.7, 0.3, 2.659, 1.3, 1126259642.413, 1.375, -1.2108]

        self.injection_parameters = {
            **dict(zip(self.search_parameter_keys, self.search_parameters)),
            **dict(zip(self.fixed_parameter_keys, self.fixed_parameters))
        }


    def get_gw_source_ifo_and_likelihood(self):


        waveform_arguments = dict(waveform_approximant=self.waveform_model,
                          reference_frequency=self.reference_frequency,
                          minimum_frequency=self.minimum_frequency)

        waveform_generator = bilby.gw.WaveformGenerator(duration=self.duration, sampling_frequency=self.sampling_frequency,
                                                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                waveform_arguments=waveform_arguments,
                                                )

        self.ifos = bilby.gw.detector.InterferometerList(self.ifos_keys)
        self.ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=self.sampling_frequency, duration=self.duration,
                                                        start_time=self.injection_parameters['geocent_time'] - 3)
        self.ifos.inject_signal(waveform_generator=waveform_generator,
                        parameters=self.injection_parameters)

        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=self.ifos, 
                                                      waveform_generator=waveform_generator)

        # set the default likelihood pars so that once likelihood is called you
        # only need to update the search parameters, leaving the fixed ones as this default!
        likelihood.parameters = self.injection_parameters                                            

        return self.ifos, likelihood


    def get_log_noise_evidence(self):
        log_l = 0
        for interferometer in self.ifos:
            mask = interferometer.frequency_mask
            log_l -= bilby.gw.utils.noise_weighted_inner_product(
                interferometer.frequency_domain_strain[mask],
                interferometer.frequency_domain_strain[mask],
                interferometer.power_spectral_density_array[mask],
                self.duration) / 2
        return float(np.real(log_l))


# TODO: set up logging

# parameters
parameter_names = [["mass_1", "m_1"],
                   ["mass_2", "m_2"], 
                   ["luminosity_distance", "d_{\mathrm{L}}"],
                   ["theta_jn", "\iota"],
                   ] 
n_params = len(parameter_names)
ndim = len(parameter_names)
priors = bilby.gw.prior.BBHPriorDict().from_json("examples/data/bilby_prior.json")
save_chains = False # if false delete all chains after running example
save_plots = True




chain_dir ="examples/data/gw_chains/"

if not os.path.exists(chain_dir):
    os.makedirs(chain_dir)


multinest_run_name = 'gw_multinest'
multinest_prefix = os.path.join(chain_dir, multinest_run_name)
np.savetxt(multinest_prefix+".paramnames", parameter_names, fmt="%s")

emcee_run_name = 'gw_emcee'
emcee_prefix = os.path.join(chain_dir, emcee_run_name)
np.savetxt(emcee_prefix+".paramnames", parameter_names, fmt="%s")

# after functions have been defined

gw_class = gw_likelihood_based(seed=1759265920)
ifos, likelihood = gw_class.get_gw_source_ifo_and_likelihood()
log_noise_evidence = gw_class.get_log_noise_evidence()

# Pymultinest

def multinest_prior(search_params : list):
    return priors.rescale(gw_class.search_parameter_keys, search_params)


def multinest_loglike(search_params : list):                    
    search_params_dict = dict(zip(gw_class.search_parameter_keys, search_params))
    likelihood.parameters.update(search_params_dict)
    return likelihood.log_likelihood() 

result = pymultinest.solve(LogLikelihood=multinest_loglike, 
                           Prior=multinest_prior, 
                           n_dims=ndim, 
                           outputfiles_basename=chain_dir+multinest_run_name, 
                           verbose=False,
                           )

multinest_log_bayes_factor = result["logZ"] - log_noise_evidence
print(f"PyMultinest logBF = {multinest_log_bayes_factor}") 

# TODO: make logging





def emcee_loglike(search_params_dict : dict):
    if not np.all([priors[key].is_in_prior_range(search_params_dict[key]) for key in gw_class.search_parameter_keys]):
        return np.nan_to_num(-np.inf)
    else: 
        likelihood.parameters.update(search_params_dict)
        return likelihood.log_likelihood() 



def emcee_posterior(search_params : list,):

    search_params_dict = dict(zip(gw_class.search_parameter_keys, search_params))
    logprior = priors.ln_prob(search_params_dict)
    if np.isinf(logprior):
        return np.nan_to_num(-np.inf)
    else:
        loglike = emcee_loglike(search_params_dict)
        return loglike + logprior

nwalkers = 100
nsamples = 1300
nburn=300

FIDUCIAL = np.array([ 30.,   30.,  1500., 0.5])
EPSILON  = np.array([ 1.,    1.,    100., 1E-2])
pos = [FIDUCIAL + EPSILON * np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_posterior)


sampler.run_mcmc(pos, nsamples)
samples = np.ascontiguousarray(sampler.chain[:,nburn:,:])
lnprob = np.ascontiguousarray(sampler.lnprobability[:,nburn:])

emcee_samples_for_getdist = np.hstack([np.ones((nwalkers*(nsamples-nburn), 1)), lnprob.reshape(-1, 1), samples.reshape(-1, 4)])
np.savetxt(emcee_prefix+".txt", emcee_samples_for_getdist)

chains = hm.Chains(ndim)
chains.add_chains_3d(samples, lnprob)
chains_train, chains_infer = hm.utils.split_data(chains, training_proportion=0.2)

domain_hs = [[1E-1,5E0]]
model_hs = hm.model.HyperSphere(ndim, domain_hs)
fit_success_hs, objective_hs = model_hs.fit(chains_train.samples, chains_train.ln_posterior)  

centre_hs = [centre_val for centre_val in list(model_hs.centre)]
radius_hs = model_hs.R # is it a problem that 

print('HyperSphere Centre = {} \nwith Radius = {} \nfrom domain_hs = {} \n------'.format(
    centre_hs,radius_hs,domain_hs))

ev_hs = hm.Evidence(chains_infer.nchains, model_hs)
ev_hs.add_chains(chains_infer)

ln_evidence_hs, ln_evidence_std_hs = ev_hs.compute_ln_evidence()
print('HyperSphere log_z = {} ± {}'.format(ln_evidence_hs, ln_evidence_std_hs))   
print(ev_hs.nsamples_eff_per_chain)

print(f"HyperSphere lnBF = {ln_evidence_hs - log_noise_evidence}")

# TODO: Get error (convert to loglin space then get %tage and then add to BF error to propagate through)


# PLOT both posteriors:
g=plots.get_subplot_plotter(chain_dir=os.path.abspath(chain_dir))
roots = [multinest_run_name, emcee_run_name]
g.triangle_plot(roots, gw_class.search_parameter_keys, filled=True, markers=gw_class.search_parameters)
g.export('examples/plots/gw_posteriors.pdf')

# TODO: clear chains dir after run by default
# TODO: convert to function then do if name==main etc.
# TODO: add heavy logging
# TODO: add debug level diagnostics
# TODO: time it roughly

# TODO: change to KDE and do model sampling

# if not save_chains:
#     os.rmdir(chain_dir) # dont do this

# TODO: rewrite so that it checks if the folder a) exists b) is empty, rethink it all tbh













       























