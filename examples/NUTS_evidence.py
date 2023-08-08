import numpy as np
import sys
import harmonic as hm
sys.path.append(".")
from harmonic import model_nf, flows
sys.path.append("examples")
import utils
import cloudpickle
import matplotlib.pyplot as plt

samples_infer = np.load('examples/data/NUTS/nuts_90ksamples_37params_test.npy')
logprob_infer = np.load('examples/data/NUTS/nuts_90ksamples_37params_test_logprob.npy')

nchains = samples_infer.shape[0]
nsamples = samples_infer.shape[1]
ndim = samples_infer.shape[-1]

var_scale = 0.7


flow_name = 'splines'
load_lab = 'splines_13l_3e_s'
model_file = 'examples/data/NUTS/model_' + load_lab
save_lab = load_lab + '_' + str(var_scale)


file = open(model_file,"rb")
model = cloudpickle.load(file)
file.close()


print("Sampling from flow...")
num_samp = samples_infer.shape[1]
samps_compressed = np.array(model.sample(num_samp, var_scale=var_scale))

# Check model loaded in correctly
plot = False
if plot:
    print('Plotting...')
    plotdim = 7
    utils.plot_getdist_compare(samples_infer[0][:,:plotdim], samps_compressed[:,:plotdim], fontsize= 2, legend_fontsize=12.5)
    plt.savefig('examples/plots/' + save_lab + '_corner_all_T' +str(var_scale) + '_cosmo_test.png', bbox_inches='tight', dpi=300)

    plotdim = ndim
    utils.plot_getdist_compare(samples_infer[0][:,:plotdim], samps_compressed[:,:plotdim], fontsize= 2, legend_fontsize=12.5)
    plt.savefig('examples/plots/' + save_lab + '_corner_all_T' +str(var_scale) + '_test.png', bbox_inches='tight', dpi=300)


print('Configure chains...')
"""
Configure chains for the cross-validation stage.
"""
samples_infer_C = samples_infer.astype('double')
logprob_infer_C = logprob_infer.astype('double')
chains = hm.Chains(ndim)
chains.add_chains_3d(samples_infer_C, logprob_infer_C)
print("nchains ", chains.nchains)
chains.split_into_blocks(nblocks=nchains*4)
print("nchains ", chains.nchains)


print('Compute evidence...')
"""
Instantiates the evidence class with a given model. Adds some chains and 
computes the log-space evidence (marginal likelihood).
"""
ev = hm.Evidence(chains.nchains, model)
ev.add_chains(chains, bulk_calc=True, var_scale=var_scale)
ln_evidence, ln_evidence_std = ev.compute_ln_evidence()
evidence_std_log_space = np.log(np.exp(ln_evidence) + np.exp(ln_evidence_std)) - ln_evidence

#===========================================================================
# Display evidence results 
#===========================================================================
print('ln_evidence_std = {} '.format(ln_evidence_std))
print('ln_evidence = {} +/- {}'.format(ln_evidence, evidence_std_log_space))
print('kurtosis = {}'.format(ev.kurtosis))
print('sqrt( 2/(n_eff-1) ) = {}'.format(np.sqrt(2.0/(ev.n_eff-1))))
check = np.exp(0.5 * ev.ln_evidence_inv_var_var - ev.ln_evidence_inv_var)
print('sqrt(evidence_inv_var_var) / evidence_inv_var = {}'.format(check))