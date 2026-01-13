import jax
import os
import jax.numpy as np
import jax_cosmo as jc
import numpy as onp
from astropy.io import fits
import scipy

def symmetrized_matrix(U):
    u"""Return a new matrix like `U`, but with upper-triangle elements copied to lower-triangle ones."""
    M = U.copy()
    inds = onp.triu_indices_from(M,k=1)
    M[(inds[1], inds[0])] = M[inds]
    return M



def symmetric_positive_definite_inverse(M):
    u"""Compute the inverse of a symmetric positive definite matrix `M`.

    A :class:`ValueError` will be thrown if the computation cannot be
    completed.

    """
    import scipy.linalg
    U,status = scipy.linalg.lapack.dpotrf(M)
    if status != 0:
        raise ValueError("Non-symmetric positive definite matrix")
    M,status = scipy.linalg.lapack.dpotri(U)
    if status != 0:
        raise ValueError("Error in Cholesky factorization")
    M = symmetrized_matrix(M)
    return M

def get_data():
    # Let's grab the data file
    if not os.path.isfile('2pt_NG_mcal_1110.fits'):
        os.system("wget http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits")

    nz_source=fits.getdata('examples/data/2pt_NG_mcal_1110.fits', 6)
    nz_lens=fits.getdata('examples/data/2pt_NG_mcal_1110.fits', 7)

    return nz_source, nz_lens



# First, let's define a function to go to and from a 1d parameter vector
def get_params_vec(cosmo, m, dz, ia, bias):
    m1, m2, m3, m4 = m
    dz1, dz2, dz3, dz4 = dz
    A, eta = ia
    b1, b2, b3, b4, b5 = bias
    return np.array([ 
        # Cosmological parameters
        cosmo.sigma8, cosmo.Omega_c, cosmo.Omega_b,
        cosmo.h, cosmo.n_s, #cosmo.w0,
        # Shear systematics
        m1, m2, m3, m4,
        # Photoz systematics
        dz1, dz2, dz3, dz4,
        # IA model
        A, eta,
        # linear galaxy bias
        b1, b2, b3, b4, b5
    ])
    
def unpack_params_vec(params):
    # Retrieve cosmology
    cosmo = jc.Cosmology(sigma8=params[0], Omega_c=params[1], Omega_b=params[2],
                         h=params[3], n_s=params[4], w0=-1,#params[5],
                         Omega_k=0., wa=0.)
    #m1,m2,m3,m4 = params[6:10]
    #dz1,dz2,dz3,dz4 = params[10:14]
    #A = params[14]
    #eta = params[15]
    #bias = params[16:21]

    m1,m2,m3,m4 = params[5:9]
    dz1,dz2,dz3,dz4 = params[9:13]
    A = params[13]
    eta = params[14]
    bias = params[15:20]
    
    return cosmo, [m1,m2,m3,m4], [dz1,dz2,dz3,dz4], [A, eta], bias


def theory_components(nzs_s, nzs_l, p):
    cosmo, m, dz, (A, eta), bias = unpack_params_vec(p) 
    # Build source nz with redshift systematic bias
    nzs_s_sys = [jc.redshift.systematic_shift(nzi, dzi) 
                for nzi, dzi in zip(nzs_s, dz)]

    # Define IA model, z0 is fixed
    b_ia = jc.bias.des_y1_ia_bias(A, eta, 0.62)
    # Bias for the lenses
    b = [jc.bias.constant_linear_bias(bi) for bi in bias] 

    # Define the lensing and number counts probe
    probes = [jc.probes.WeakLensing(nzs_s_sys, 
                                    ia_bias=b_ia,
                                    multiplicative_bias=m),
             jc.probes.NumberCounts(nzs_l, b)]
    return cosmo, probes



# These 
@jax.jit
def theory_cl(p, nzs_s, nzs_l, ell):
    cosmo, probes = theory_components(nzs_s, nzs_l, p)
    cl = jc.angular_cl.angular_cl(cosmo, ell, probes)
    return cl

# These 
@jax.jit
def theory_mean(p, nzs_s, nzs_l, ell):
    cl = theory_cl(p, nzs_s, nzs_l, ell)
    return cl.flatten()

theory_jacobian = jax.jacfwd(jax.jit(theory_mean))



@jax.jit
def theory_cov(p, nzs_s, nzs_l, ell):
    cosmo, probes = theory_components(nzs_s, nzs_l, p)
    cl_signal = jc.angular_cl.angular_cl(cosmo, ell, probes)
    cl_noise = jc.angular_cl.noise_cl(ell, probes)
    cov = jc.angular_cl.gaussian_cl_covariance(ell, probes, cl_signal, cl_noise, f_sky=0.25, sparse=False)
    return cov



class MockY1Likelihood:

    def __init__(self):
        nz_source, nz_lens = get_data()
        # This is the effective number of sources from the cosmic shear paper
        self.neff_s = [1.47, 1.46, 1.50, 0.73]

        self.nzs_s = [jc.redshift.kde_nz(nz_source['Z_MID'].astype('float32'),
                                    nz_source['BIN%d'%i].astype('float32'), 
                                    bw=0.01,
                                    gals_per_arcmin2=self.neff_s[i-1])
                   for i in range(1,5)]

        self.nzs_l = [jc.redshift.kde_nz(nz_lens['Z_MID'].astype('float32'),
                                      nz_lens['BIN%d'%i].astype('float32'), bw=0.01)
                   for i in range(1,6)]    

        # Define some ell range
        self.ell = np.logspace(1, 3)
        fid_cosmo = jc.Cosmology(sigma8=0.801,
                                  Omega_c=0.2545,
                                  Omega_b=0.0485,
                                  h=0.682,
                                  n_s=0.971,
                                  w0=-1., 
                                  Omega_k=0., wa=0.)
        self.fid_params  = get_params_vec(fid_cosmo, 
                                          [0., 0., 0., 0.],
                                          [0., 0., 0., 0.],
                                          [0.5, 0.],
                                          [1.2, 1.4, 1.6, 1.8, 2.0])


        # Set data as fiducial theory
        print("Initializing mock likelihood")
        self.args = [self.nzs_s, self.nzs_l, self.ell]
        self.data_mean = theory_mean(self.fid_params, self.nzs_s, self.nzs_l, self.ell)
        self.data_cov = theory_cov(self.fid_params, *self.args)
        self.data_inv_cov = symmetric_positive_definite_inverse(self.data_cov)

    def fisher_matrix(self, p0):
        j = theory_jacobian(p0, *self.args)
        j = onp.array(j)
        F = onp.einsum('ia,ij,jb->ab', j, self.data_inv_cov, j)
        F = 0.5*(F + F.T)

        return F

    def cov_estimate(self):
        F = self.fisher_matrix(self.fid_params)
        C = symmetric_positive_definite_inverse(F)
        return C


    def priors(self, p):
        #priors - index, mean, std. dev
        prior_values = [
            (5, 0.012, 0.023),  #m1
            (6, 0.012, 0.023),  #m2
            (7, 0.012, 0.023),  #m3
            (8, 0.012, 0.023),  #m4
            (9, -0.001, 0.016),   #dz1
            (10, -0.019, 0.013),   #dz2
            (11,0.009, 0.011),   #dz3
            (12, -0.018, 0.022),   #dz4
        ]

        logpi = 0.0
        dlogpi_dp = onp.zeros_like(p)
        for i, mu_i, sigma_i in prior_values:
            logpi += -0.5 * (p[i] - mu_i)**2 / sigma_i**2
            dlogpi_dp[i] = - (p[i] - mu_i) / sigma_i**2
        return logpi, dlogpi_dp

    def posterior(self, p):
        cl = theory_mean(p, *self.args)
        d = cl - self.data_mean
        logL = -0.5 * d @ self.data_inv_cov @ d
        logPi, _ = self.priors(p)
        logP = logL + logPi
        return logP

    # returns -posterior P and -dL/dP
    def posterior_and_gradient(self, p):
        # theory C_ell prediction
        cl = theory_mean(p, *self.args)
        # d C_ell / d p
        j = theory_jacobian(p, *self.args).T
        d = cl - self.data_mean
        dlogL_dCl = -self.data_inv_cov @ d
        logL = 0.5 * d @ dlogL_dCl
        dlogL_dp = j @ dlogL_dCl
        
        # Add Gaussian priors.
        # Can't use += because of JAX
        logPi, dlogPi_dp = self.priors(p)
        logP = logL + logPi
        dlogP_dp = dlogL_dp + dlogPi_dp

        # convert back to regular numpy arrays
        return onp.array(logP), onp.array(dlogP_dp)


def plot_contours(fisher, pos, i, j, nstd=1., ax=None, resize=False, **kwargs):
    """
    Plot 2D parameter contours given a Hessian matrix of the likelihood
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
      
    def eigsorted(cov):
        vals, vecs = onp.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    cov = onp.linalg.inv(fisher)
    sigma_marg = lambda i: np.sqrt(cov[i, i])

    if ax is None:
        ax = plt.gca()

    # Extracts the block we are interested in
    cov = cov[:,[i,j]][[i,j],:]
    vals, vecs = eigsorted(cov)
    theta = onp.degrees(onp.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * onp.sqrt(vals)
    xy = [pos[i], pos[j]]
    ellip = Ellipse(xy=xy, width=width,
                  height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    if resize:
        sz = max(width, height)
        s1 = 1.5*nstd*sigma_marg(i)
        s2 = 1.5*nstd*sigma_marg(j)
        ax.set_xlim(pos[i] - s1, pos[i] + s1)
        ax.set_ylim(pos[j] - s2, pos[j] + s2)
    plt.draw()
    return ellip

def test():
    import matplotlib.pyplot as plt
    L = MockY1Likelihood()
    F = L.fisher_matrix(L.fid_params)
    plot_contours(F, L.fid_params, 0, 1, resize=True)
    print("Saving fisher.png")
    plt.savefig('fisher.png')
    plt.close()


    p0 = onp.array(L.fid_params)
    n = 20
    ll0 = L.posterior(p0)
    Sigma8 = onp.linspace(0.79, 0.81, n)
    ll = onp.zeros(n)
    for i, sigma8 in enumerate(Sigma8):
        print(i, sigma8)
        p0[0] = sigma8
        ll[i] = L.posterior(p0)
    plt.plot(Sigma8, ll)

    print("Saving logp.png")
    plt.savefig('logp.png')
    plt.close()


if __name__ == '__main__':
    test()
