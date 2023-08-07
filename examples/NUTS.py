import numpy as np
import sys
sys.path.append(".")
from harmonic import model_nf, flows
sys.path.append("examples")
import utils
import matplotlib.pyplot as plt

savefigs = True
plot_corner = True
flow_name = "nvp"
example_name = "NUTS"

samples_train = np.load('examples/data/NUTS/nuts_90ksamples_37params_train.npy')[0]

#Flow and training parameters
epochs_num = 500
var_scale = 0.8
standardize = True

ndim = samples_train.shape[1]
n_scaled = 4
n_unscaled = 3
learning_rate = 0.001
momentum = 0.9



print('Fit model for {} epochs...'.format(epochs_num))
model = model_nf.RealNVPModel(ndim, n_scaled_layers=n_scaled, n_unscaled_layers=n_unscaled, learning_rate = learning_rate, standardize=standardize)
#model = model_nf.RQSplineFlow(ndim)
model.fit(samples_train, epochs=epochs_num) 

print("Sampling from flow...")
num_samp = samples_train.shape[0]
samps_compressed = np.array(model.sample(num_samp, var_scale=var_scale))

print('Plotting...')

plot_posterior = True
if plot_posterior:
    utils.plot_getdist(samples_train.reshape((-1, ndim)))
    if savefigs:
        plt.savefig('examples/plots/' + flow_name + '_' + example_name + '_getdist.png', bbox_inches='tight', dpi=300)  

    utils.plot_getdist_compare(samples_train, samps_compressed[:,:ndim], fontsize= 2, legend_fontsize=12.5)
    if savefigs:
        plt.savefig('examples/plots/' + flow_name + '_' + example_name + '_corner_all_T' +str(var_scale) + '.png', bbox_inches='tight', dpi=300)

    utils.plot_getdist(samps_compressed.reshape((-1, ndim)))
    if savefigs:
        plt.savefig('examples/plots/' + flow_name + '_' + example_name + '_flow.png', bbox_inches='tight', dpi=300)  


plot_cosmo_posterior = True
plotdim = 7

if plot_cosmo_posterior:
    utils.plot_getdist(samples_train[:,:plotdim].reshape((-1, plotdim)))
    if savefigs:
        plt.savefig('examples/plots/' + flow_name + '_' + example_name + '_cosmo.png', bbox_inches='tight', dpi=300)  

    utils.plot_getdist_compare(samples_train[:,:plotdim], samps_compressed[:,:plotdim], fontsize= 2, legend_fontsize=12.5)
    if savefigs:
        plt.savefig('examples/plots/' + flow_name + '_' + example_name + '_corner_all_T' +str(var_scale) + '_cosmo.png', bbox_inches='tight', dpi=300)

    utils.plot_getdist(samps_compressed[:,:plotdim].reshape((-1, plotdim)))
    if savefigs:
        plt.savefig('examples/plots/' + flow_name + '_' + example_name + '_flow_cosmo.png', bbox_inches='tight', dpi=300)  

plt.show(block=False)
