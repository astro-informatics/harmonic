import numpy as np 
# import matplotlib
import matplotlib.pyplot as plt
import argparse
# import gaussian_nonsym_multirealisations as results
import sys
sys.path.append("examples")
import utils

# Run by:
# ~/Src/harmonic   review  python examples/plot_example.py <>

parser = argparse.ArgumentParser("Create violin plot of inverse evidences" +
    "from many realisations")
parser.add_argument('filename_realisations', metavar='filename_realisations', 
                    type=str, 
                    help='Name of file containing realisations')
parser.add_argument('filename_analytic', metavar='filename_analytic', 
                    type=str, 
                    help='Name of file containing analytic inverse variance')
args = parser.parse_args()








evidence_inv_summary = np.loadtxt(args.filename_realisations)
evidence_inv_realisations = evidence_inv_summary[:,0]
evidence_inv_var_realisations = evidence_inv_summary[:,1]
evidence_inv_var_var_realisations = evidence_inv_summary[:,2]


# evidence_inv_mean = np.mean(evidence_inv_summary[:,0])
# evidence_inv_std_measured = np.std(evidence_inv_summary[:,0])
# evidence_inv_var_mean_est = np.mean(evidence_inv_summary[:,1])
# evidence_inv_var_std_meas = np.std(evidence_inv_summary[:,1])
# evidence_inv_std_mean_est = np.sqrt(np.mean(evidence_inv_summary[:,1]))
# evidence_inv_var_var_mean = np.mean(evidence_inv_summary[:,2])
# 
# 
evidence_inv_analytic = np.loadtxt(args.filename_analytic)
# evidence_inv_analytic = evidence_inv_analytic_summary[0]

print("evidence_inv_analytic = {}".format(evidence_inv_analytic))


# rho_mean         = np.mean(rho_array[:,0])
# rho_std_measured = np.std(rho_array[:,0])
# rho_var_mean_est = np.mean(rho_array[:,1])
# rho_var_std_meas = np.std(rho_array[:,1])
# rho_std_mean_est = np.sqrt(np.mean(rho_array[:,1]))
# rho_var_var_mean = np.mean(rho_array[:,2])
# 
# plt.figure()
# plt.plot(np.arange(3),np.zeros(3)+np.exp(results.ln_rho))
# plt.plot(np.zeros(results.n_real)+1,rho_array[:,0],'k+')
# plt.errorbar(np.zeros(1)+0.9,rho_mean,yerr=rho_std_measured, fmt='--o')
# plt.errorbar(np.zeros(1)+1.1,rho_mean,yerr=rho_std_mean_est, fmt='--o')
# 
# plt.figure()
# # plt.plot(np.arange(3),np.ones(3))
# plt.errorbar(np.zeros(1)+0.9,rho_var_mean_est,yerr=rho_var_std_meas, fmt='--o')
# plt.errorbar(np.zeros(1)+1.1,rho_var_mean_est,yerr=np.sqrt(rho_var_var_mean), fmt='--o')






# print(plt.rcParams.get('figure.figsize'))
# 
# matplotlib.rcParams.update({'font.size': 20})
# 
# plot_aspect_ratio = 1.33
# plot_x_size = 9
# 
# fig, ax = plt.subplots(figsize=(plot_x_size, plot_x_size/plot_aspect_ratio))
# ax.violinplot(rho_array[:,0], showmeans=False, showmedians=False,
#         showextrema=True, bw_method=1.0)
# plt.plot(np.arange(4),np.zeros(4)+np.exp(results.ln_rho), 'r--')     
# ax.text(1.73, np.exp(results.ln_rho)+0.00003, r'Analytic', color='red')
# 
# plt.errorbar(np.zeros(1)+1.0,rho_mean,yerr=rho_std_measured, fmt='--o', color='C4', capsize=7, capthick=3, linewidth=3, elinewidth=3)  
# plt.errorbar(np.zeros(1)+1.5,rho_mean,yerr=rho_std_mean_est, fmt='--o', color='C2', capsize=7, capthick=3, linewidth=3, elinewidth=3)
# 
# ax.set_xlim([0.5, 2.0])
# ax.set_ylim([0.0077, 0.0087])
# 
# ax.get_xaxis().set_tick_params(direction='out')
# ax.xaxis.set_ticks_position('bottom')
# ax.set_xticks([1.0, 1.5])
# ax.set_xticklabels(['Measured', 'Estimated'])
# 
# ax.set_ylabel(r'Inverse evidence ($\rho$)')
    
  
        
        
        
plt.rcParams.update({'font.size': 20})
    
ax = utils.plot_realisations(mc_estimates=evidence_inv_realisations,  
                             std_estimated=np.sqrt(np.mean(evidence_inv_var_realisations)),
                             analytic_val=evidence_inv_analytic,
                             analytic_text=r'Truth')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))                             
ax.set_ylabel(r'Inverse evidence ($\rho$)')
# ax.set_ylim([0.0077, 0.0087])
    
plt.savefig('./plots/temp2.pdf',
            bbox_inches='tight')      
        
        
        
ax = utils.plot_realisations(mc_estimates=evidence_inv_var_realisations, 
                             std_estimated=np.sqrt(np.mean(evidence_inv_var_var_realisations)))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))        
ax.set_ylabel(r'Inverse evidence variance ($\sigma^2$)')        
# ax.set_ylim([-0.5E-8, 1.5E-8])
    
        
        
        
plt.show(block=False)

input("\nPress Enter to continue...")
