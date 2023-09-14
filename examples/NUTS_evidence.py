import numpy as np
import sys
import harmonic as hm
sys.path.append(".")
from harmonic import model_nf, flows
sys.path.append("examples")
import utils
import cloudpickle
import matplotlib.pyplot as plt

#samples_infer = np.load('examples/data/NUTS/nuts_90ksamples_37params_test.npy')
#logprob_infer = np.load('examples/data/NUTS/nuts_90ksamples_37params_test_logprob.npy')

train_chains = 7
samples_infer = np.load('examples/data/NUTS/nuts_samples_highprec_lcdm_15_2k.npy')[train_chains:,:,:]
logprob_infer = np.load('examples/data/NUTS/nuts_samples_highprec_lcdm_15_2k_pe.npy')[train_chains:,:]
print(samples_infer.shape, logprob_infer.shape)

nchains = samples_infer.shape[0]
nsamples = samples_infer.shape[1]
ndim = samples_infer.shape[-1]


flow_name = 'splines'
#load_lab = 'splines_8l_300e_s'
load_lab = 'splines_NUTS_highprec_13l_300e_s'
model_file = 'examples/data/NUTS/model_' + load_lab

file = open(model_file,"rb")
model = cloudpickle.load(file)
file.close()

var_scale = model.temperature
save_lab = load_lab + '_' + str(var_scale) + '_highprec'


# Check model loaded in correctly
plot = True
if plot:
    print("Sampling from flow...")
    #num_samp = samples_infer.shape[1]
    num_samp = nchains*nsamples
    samps_compressed = np.array(model.sample(num_samp))

    print('Plotting...')
    plotdim = 7
    utils.plot_getdist_compare(samples_infer.reshape((-1,ndim))[:,:plotdim], samps_compressed[:,:plotdim], fontsize= 2, legend_fontsize=12.5)
    plt.savefig('examples/plots/' + save_lab + '_corner_all_T' +str(var_scale) + '_cosmo_test.png', bbox_inches='tight', dpi=300)

    plotdim = ndim
    utils.plot_getdist_compare(samples_infer.reshape((-1,ndim))[:,:plotdim], samps_compressed[:,:plotdim], fontsize= 2, legend_fontsize=12.5)
    plt.savefig('examples/plots/' + save_lab + '_corner_all_T' +str(var_scale) + '_test.png', bbox_inches='tight', dpi=300)


print('Configure chains...')
samples_infer_C = samples_infer.astype('double')
logprob_infer_C = logprob_infer.astype('double')
chains = hm.Chains(ndim)
chains.add_chains_3d(samples_infer_C, logprob_infer_C)
print("nchains", chains.nchains)


print('Compute evidence...')
"""
Instantiates the evidence class with a given model. Adds some chains and 
computes the log-space evidence (marginal likelihood).
"""
ev = hm.Evidence(chains.nchains, model, batch_calculation = True)
ev.add_chains(chains)
ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()

#===========================================================================
# Display evidence results 
#===========================================================================
print('ln_evidence_std ', ln_evidence_std)
print('ln_inv_evidence = {} +/- {}'.format(ev.ln_evidence_inv, err_ln_inv_evidence))
print('kurtosis = {}'.format(ev.kurtosis))
print('sqrt( 2/(n_eff-1) ) = {}'.format(np.sqrt(2.0/(ev.n_eff-1))))
check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
print('sqrt(evidence_inv_var_var) / evidence_inv_var = {}'.format(check))