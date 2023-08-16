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

trunc = 11

samples_train = np.load('examples/data/NUTS/nuts_90ksamples_37params_train.npy')[0][:,:trunc]
logprob_train = np.load('examples/data/NUTS/nuts_90ksamples_37params_train_logprob.npy')

means = np.mean(samples_train, axis=0)
std = np.std(samples_train, axis = 0)
print(samples_train.shape)
#samples_train = (samples_train-means)/std

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


print(model_hs.centre, model_hs.inv_covariance)


samples_infer = np.load('examples/data/NUTS/nuts_90ksamples_37params_test.npy')[:,:,:trunc].astype('double')
logprob_infer = np.load('examples/data/NUTS/nuts_90ksamples_37params_test_logprob.npy').astype('double')

#samples_infer =  (samples_infer-means)/std

nchains = samples_infer.shape[0]
nsamp = samples_infer.shape[1]

chains = hm.Chains(ndim)
chains.add_chains_3d(samples_infer, logprob_infer)

#stand_samples_test = (samples_infer.reshape((nchains*nsamp, ndim)) - model_hs.centre)*model_hs.inv_covariance**0.5
#stand_samples_train = (samples_train - model_hs.centre)*model_hs.inv_covariance**0.5


"""for i in range(ndim):
    plt.hist(stand_samples_test[:,i], label="Infer")
    plt.hist(stand_samples_train[:,i], label = "Train")
    plt.title("param " + str(i) + "STD model" + str(model_hs.inv_covariance[i]**(-0.5)))
    plt.legend()
    plt.savefig("hist_"+str(i)+".png")
    #plt.show()"""

"""
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



par_names = np.arange(37)
print("Plotting...")
g_hs = plots.getSubplotPlotter()
g_hs.triangle_plot([samplesMC,samples_model_MC_hs], \
    legend_labels=["LVC", "Learnt Target"], names=par_names, filled=True, contour_colors=['red','tab:blue'], line_args=[{'ls':'-', 'color':'red'}, {'ls':'--', 'color':'blue'}])
plt.legend()
plt.savefig("examples/plots/hs.png")
plt.show()

g_hs = plots.getSubplotPlotter()
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
#std_ln_evidence_hs = np.log(np.exp(ln_evidence_hs) + np.exp(ln_evidence_std_hs))- ln_evidence_hs # convert to log space variance
ln_inv_evidence_hs = ev_hs.ln_evidence_inv
std_ln_inv_evidence_hs = ev_hs.compute_ln_inv_evidence_errors()
print('log inv evidence (harmonic) = {} ± {}'.format(ln_inv_evidence_hs, std_ln_inv_evidence_hs))