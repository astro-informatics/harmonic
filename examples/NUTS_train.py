import numpy as np
import sys
sys.path.append(".")
from harmonic import model_nf, flows
sys.path.append("examples")
import utils
import matplotlib.pyplot as plt

savefigs = False
#flow_name = "nvp" 
flow_name = 'splines'
example_name = "NUTS"

samples_train = np.load('examples/data/NUTS/nuts_90ksamples_37params_train.npy')[0]

#Flow and training parameters
epochs_num = 300
var_scale = 0.8
standardize = True



if standardize:
    stand_lab = 's'
else:
    stand_lab = 'ns'

ndim = samples_train.shape[1]

# NVP params
n_scaled = 13
n_unscaled = 6

#Spline params
n_layers = 8
n_bins = 8
hidden_size = [64, 64]
spline_range = (-10.0, 10.0)

# Optimizer params
learning_rate = 0.001
momentum = 0.9


if flow_name == 'nvp':
    model = model_nf.RealNVPModel(ndim, n_scaled_layers=n_scaled, n_unscaled_layers=n_unscaled, learning_rate = learning_rate, standardize=standardize, temperature=var_scale)
    save_lab = flow_name + '_' + example_name + '_' + str(n_scaled) + '_' + str(n_unscaled)  + 'l_' + str(epochs_num) + 'e_' + stand_lab
if flow_name == 'splines':    
    model = model_nf.RQSplineFlow(ndim, n_layers = n_layers, n_bins = n_bins, hidden_size = hidden_size, spline_range = spline_range, standardize = standardize, learning_rate = learning_rate, momentum = momentum, temperature=var_scale)
    save_lab = flow_name + '_' + str(n_layers) + 'l_' + str(epochs_num) + 'e_' + stand_lab


print('Fit model for {} epochs...'.format(epochs_num))
model.fit(samples_train, epochs=epochs_num) 

model.serialize("examples/data/NUTS/model_"+save_lab)



print("Sampling from flow...")
num_samp = samples_train.shape[0]
samps_compressed = np.array(model.sample(num_samp))

print('Plotting...')
plot_training = False 
plot_flow = False

plot_posterior = False
if plot_posterior:
    if plot_training:
        utils.plot_getdist(samples_train.reshape((-1, ndim)))
        if savefigs:
            plt.savefig('examples/plots/' + save_lab + '_getdist.png', bbox_inches='tight', dpi=300)  

    utils.plot_getdist_compare(samples_train, samps_compressed[:,:ndim], fontsize= 2, legend_fontsize=12.5)
    if savefigs:
        plt.savefig('examples/plots/' + save_lab + '_corner_all_T' +str(var_scale) + '.png', bbox_inches='tight', dpi=300)

    if plot_flow:
        utils.plot_getdist(samps_compressed.reshape((-1, ndim)))
        if savefigs:
            plt.savefig('examples/plots/' + save_lab + '_flow.png', bbox_inches='tight', dpi=300)  


plot_cosmo_posterior = True
plotdim = 7

if plot_cosmo_posterior:
    if plot_training:
        utils.plot_getdist(samples_train[:,:plotdim].reshape((-1, plotdim)))
        if savefigs:
            plt.savefig('examples/plots/' + save_lab + '_cosmo.png', bbox_inches='tight', dpi=300)  

    utils.plot_getdist_compare(samples_train[:,:plotdim], samps_compressed[:,:plotdim], fontsize= 2, legend_fontsize=12.5)
    if savefigs:
        plt.savefig('examples/plots/' + save_lab + '_corner_all_T' +str(var_scale) + '_cosmo.png', bbox_inches='tight', dpi=300)

    utils.plot_getdist(samps_compressed[:,:plotdim].reshape((-1, plotdim)))
    if savefigs:
        plt.savefig('examples/plots/' + save_lab + '_flow_cosmo.png', bbox_inches='tight', dpi=300)  

plt.show()