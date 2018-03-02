import numpy as np 
import matplotlib.pyplot as plt
import argparse
import many_real_gaussian_nonsym_lowdim as results

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('base_name', metavar='base_name', type=str, 
                   help='an integer for the accumulator')

args = parser.parse_args()

rho_array = np.loadtxt(args.base_name+".dat")

print(np.exp(results.ln_rho))
print(rho_array[:,0])

rho_mean         = np.mean(rho_array[:,0])
rho_std_measured = np.std(rho_array[:,0])
rho_std_mean_est = np.sqrt(np.mean(rho_array[:,1]))


plt.plot(np.arange(3),np.zeros(3)+np.exp(results.ln_rho))
plt.plot(np.zeros(results.n_real)+1,rho_array[:,0],'k+')
plt.errorbar(np.zeros(1)+0.9,rho_mean,yerr=rho_std_measured, fmt='--o')
plt.errorbar(np.zeros(1)+1.1,rho_mean,yerr=rho_std_mean_est, fmt='--o')
plt.show()