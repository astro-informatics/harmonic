import numpy as np
import sys
sys.path.append(".")
import harmonic as hm
from harmonic import model as md
sys.path.append("examples")
import utils
import matplotlib.pyplot as plt
import emcee
from getdist import plots, MCSamples

savefigs = True
model_name = 'hs'

samples_train = np.load('examples/data/NUTS/nuts_90ksamples_37params_train.npy')[0]
logprob_train = np.load('examples/data/NUTS/nuts_90ksamples_37params_train_logprob.npy')
print(samples_train.shape, logprob_train.shape)

samples_train = np.ascontiguousarray(samples_train).astype('double')
logprob_train = np.ascontiguousarray(logprob_train).astype('double')
ndim = samples_train.shape[1]

chains_train = hm.Chains(ndim)
chains_train.add_chain(samples_train, logprob_train)

domain = [[10E-5,10]]
model_hs = md.HyperSphere(ndim, domain)
fit_success_hs, objective_hs = model_hs.fit(chains_train.samples, chains_train.ln_posterior)
centre_hs = [centre_val for centre_val in list(model_hs.centre)]
radius_hs = model_hs.R
print("Centres ", centre_hs, "radius ", radius_hs)

print("STD ",  np.std(chains_train.samples, axis=0))

#model.serialize("examples/data/NUTS/model_"+save_lab)
samples_infer = np.load('examples/data/NUTS/nuts_90ksamples_37params_test.npy').astype('double')
logprob_infer = np.load('examples/data/NUTS/nuts_90ksamples_37params_test_logprob.npy').astype('double')
print(samples_infer.shape, logprob_infer.shape)

nchains = samples_infer.shape[0]

chains = hm.Chains(ndim)
chains.add_chains_3d(samples_infer, logprob_infer)


print("Sampling...")
samplesMC = MCSamples(samples=samples_infer)
# Visualise HS model using emcee
MODEL_NWALKERS = 200
MODEL_NSAMPLES = 10000
MODEL_NBURN = 4000
INIT_SCALE = 0.005 # param set by user
model_pos_hs = (np.random.randn(MODEL_NWALKERS,ndim)*INIT_SCALE+1) * centre_hs
print(model_pos_hs)
model_sampler_hs = emcee.EnsembleSampler(MODEL_NWALKERS, ndim, \
    lambda x : model_hs.predict(x), args = [])
model_sampler_hs.run_mcmc(model_pos_hs, MODEL_NSAMPLES) 
samples_model_hs = model_sampler_hs.chain[:,MODEL_NBURN:,:]
samples_model_MC_hs = MCSamples(samples = samples_model_hs)


print(np.ascontiguousarray(samples_model_hs).shape)
print("infer ", samples_infer.shape)

"""par_names = np.arange(37)
print("Plotting...")
g_hs = plots.getSubplotPlotter()
g_hs.triangle_plot([samplesMC,samples_model_MC_hs], \
    legend_labels=["LVC", "Learnt Target"], names=par_names, filled=True, contour_colors=['red','tab:blue'], line_args=[{'ls':'-', 'color':'red'}, {'ls':'--', 'color':'blue'}])
plt.legend()
plt.savefig("examples/plots/hs.png")
plt.show()"""

"""g_hs = plots.getSubplotPlotter()
g_hs.triangle_plot(samples_model_MC_hs, names=par_names, filled=True)
plt.savefig("examples/plots/hs2.png")
plt.show()"""

for i in range(chains.samples.shape[0]):
    prediction = model_hs.predict(chains.samples[0,:])
    if not np.isinf(prediction):
        print("Prediction " , prediction)

print("Calculating evidence...")
# estimate log evidence
ev_hs = hm.Evidence(nchains, model_hs)
ev_hs.add_chains(chains)
ln_evidence_hs, ln_evidence_std_hs = ev_hs.compute_ln_evidence()
std_ln_evidence_hs = np.log(np.exp(ln_evidence_hs) + np.exp(ln_evidence_std_hs))- ln_evidence_hs # convert to log space variance

# compare to bilby result
print('log evidence (harmonic) = {} ± {}'.format(ln_evidence_hs, std_ln_evidence_hs))