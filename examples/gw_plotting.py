import numpy as np
import bilby
import emcee
import matplotlib.pyplot as plt
import harmonic as hm
import os
import pymultinest
from getdist import plots

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


gw_class = gw_likelihood_based(seed=1759265920)
ifos, likelihood = gw_class.get_gw_source_ifo_and_likelihood()
log_noise_evidence = gw_class.get_log_noise_evidence()

chain_dir = os.path.abspath("examples/data/gw_chains")
g=plots.get_subplot_plotter(chain_dir=chain_dir)
roots = ['gw_multinest','gw_emcee']
g.triangle_plot(roots, gw_class.search_parameter_keys, filled=True, markers=gw_class.search_parameters)
g.export('gw.pdf')

