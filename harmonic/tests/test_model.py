import pytest
import harmonic.model as md
import numpy as np
import matplotlib.pyplot as plt

def test_hyper_sphere_constructor():

    with pytest.raises(ValueError):
        sphere = md.HyperSphere(2, [np.array([0.5,1.5])], hyper_parameters=[5])
    with pytest.raises(ValueError):
        sphere = md.HyperSphere(2, [np.array([0.5,1.5]),np.array([0.5,1.5])])
    with pytest.raises(ValueError):
        sphere = md.HyperSphere(0, [np.array([0.5,1.5])])

    ndim   = 3
    domain = [np.array([0.5,1.5])]
    sphere = md.HyperSphere(ndim, domain)

    assert sphere.ndim               == ndim
    assert sphere.R_domain.shape[0]  == 2
    assert sphere.R_domain[0]        == pytest.approx(0.5)
    assert sphere.R_domain[1]        == pytest.approx(1.5)
    assert sphere.R                  == pytest.approx(1.0)
    assert sphere.centre_set         == False
    assert sphere.inv_covariance_set == False
    assert sphere.fitted             == False

    for i_dim in range(ndim):
        assert sphere.centre[i_dim]        == pytest.approx(0.0)
        assert sphere.inv_covariance[i_dim] == pytest.approx(1.0)


def test_hyper_sphere_set_sphere_centre_and_shape():

    ndim   = 3
    domain = [np.array([0.5,1.5])]

    sphere = md.HyperSphere(ndim, domain)

    with pytest.raises(ValueError):
        sphere.set_centre(np.array([0.0,0.0]))
    with pytest.raises(ValueError):
        sphere.set_centre(np.array([0.0,0.0,np.nan]))

    with pytest.raises(ValueError):
        sphere.set_inv_covariance(np.array([0.0,0.0]))
    with pytest.raises(ValueError):
        sphere.set_inv_covariance(np.array([0.0,0.0,np.nan]))
    with pytest.raises(ValueError):
        sphere.set_inv_covariance(np.array([0.0,0.0,-1.0]))

    sphere.set_centre(np.array([1.5,3.0,-2.0]))
    assert sphere.centre[0]  == pytest.approx(1.5)
    assert sphere.centre[1]  == pytest.approx(3.0)
    assert sphere.centre[2]  == pytest.approx(-2.0)
    assert sphere.centre_set == True

    sphere.set_inv_covariance(np.array([2.5,4.0,1.0]))
    assert sphere.inv_covariance[0]  == pytest.approx(2.5)
    assert sphere.inv_covariance[1]  == pytest.approx(4.0)
    assert sphere.inv_covariance[2]  == pytest.approx(1.0)
    assert sphere.inv_covariance_set == True


def test_hyper_sphere_set_radius_and_precompute_values():

    ndim   = 2
    domain = [np.array([0.5,1.5])]
    sphere = md.HyperSphere(ndim, domain)

    with pytest.raises(ValueError):
        sphere.set_R(np.nan)
    with pytest.raises(ValueError):
        sphere.set_R(-0.5)

    sphere.set_R(4.0)

    assert sphere.R                  == pytest.approx(4.0)
    assert sphere.R_squared          == pytest.approx(16.0)
    assert sphere.ln_one_over_volume == pytest.approx(-np.log(16.0*np.pi))

    ndim   = 6
    domain = [np.array([0.5,1.5])]
    sphere = md.HyperSphere(ndim, domain)

    sphere.set_R(2.0)

    assert sphere.R                  == pytest.approx(2.0)
    assert sphere.R_squared          == pytest.approx(4.0)
    assert sphere.ln_one_over_volume == pytest.approx(-5.801314)
    # 5.801314 taken from wolfram alpha log(volume) of 6D r=2 sphere

    sphere.set_inv_covariance(np.full((ndim),2.0))

    assert sphere.ln_one_over_volume == pytest.approx(-5.801314-3*np.log(0.5))
    # 5.801314 taken from wolfram alpha volume of 6D r=2 sphere


def test_hyper_sphere_predict():

    ndim   = 6
    domain = [np.array([0.5,1.5])]
    sphere = md.HyperSphere(ndim, domain)
    
    assert sphere.predict(np.zeros((ndim))) == pytest.approx(-5.801314+6*np.log(2.0))
    # 5.801314 taken from wolfram alpha log(volume) of 6D r=2 sphere

    sphere.set_R(4.0)

    assert sphere.predict(np.zeros((ndim))) \
        == pytest.approx(-5.801314+6*np.log(0.5))
    # 5.801314 taken from wolfram alpha log(volume) of 6D r=2 sphere
    assert sphere.predict(np.full((ndim),4.0)) == -np.inf
    assert np.exp(sphere.predict(np.full((ndim),4.0))+1E99) \
        == pytest.approx(0.0)

    x    = np.zeros((ndim))
    x[4] = 4.0001
    assert sphere.predict(x) == -np.inf
    x[4] = 3.9999
    assert sphere.predict(x) == pytest.approx(-5.801314+6*np.log(0.5))
    # 5.801314 taken from wolfram alpha log(volume) of 6D r=2 sphere

    inv_covariance  = np.ones((ndim))*4
    sphere.set_inv_covariance(inv_covariance)   
    x[4] = 2.0001
    assert sphere.predict(x) == -np.inf
    x[4] = 1.9999
    assert sphere.predict(x) == pytest.approx(-5.801314+6*np.log(0.5)+6*np.log(2.0))
    # 5.801314 taken from wolfram alpha log(volume) of 6D r=2 sphere

    centre  = np.array([0., 0., 0., 0., 200., 0.])
    sphere.set_centre(centre)
    x[4] = 200.0 + 2.0001
    assert sphere.predict(x) == -np.inf
    x[4] = 200.0 + 1.9999
    assert sphere.predict(x) == pytest.approx(-5.801314+6*np.log(0.5)+6*np.log(2.0))
    # 5.801314 taken from wolfram alpha log(volume) of 6D r=2 sphere


def test_hyper_sphere_fit():

    ndim = 2
    domain = [np.array([1E-1,1E1])]

    sphere = md.HyperSphere(ndim, domain)

    nsamples = 10000

    with pytest.raises(ValueError):
        sphere.fit(np.ones((nsamples,ndim+1)),np.ones(nsamples))
    with pytest.raises(ValueError):
        sphere.fit(np.ones((nsamples,ndim)),np.ones(nsamples+1))

    X = np.ones((nsamples,ndim))
    Y = np.ones(nsamples)
    
    X[0,0] = np.nan
    with pytest.raises(ValueError):
        sphere.fit(X,Y)

    np.random.seed(30)
    X = np.random.randn(nsamples,ndim)
    Y = -np.sum(X*X,axis=1)/2.0

    success, _ = sphere.fit(X, Y)
    assert success == True
    assert sphere.R         == pytest.approx(1.910259417) 
    # 1.910259417 is the numerical value when first implemented (and tested) 
    # and is specified here as a regression test.
         
    assert sphere.fitted    == True

    np.random.seed(30)
    X = np.random.randn(nsamples,ndim)+1.0
    Y = -np.sum((X-1.0)**2,axis=1)/2.0

    del sphere
    sphere = md.HyperSphere(ndim, domain)
    success, _ = sphere.fit(X, Y)
    assert success == True
    assert sphere.R         == pytest.approx(1.910259417)
    # 1.910259417 is the numerical value when first implemented (and tested) 
    # and is specified here as a regression test.

    np.random.seed(30)
    X = np.random.randn(nsamples,ndim)
    X[:,1] = X[:,1]*0.5
    Y = -X[:,0]*X[:,0]/2.0 - X[:,1]*X[:,1]/(2.0*0.25)

    del sphere
    sphere = md.HyperSphere(ndim, domain)
    success, _ = sphere.fit(X, Y)
    assert success == True
    assert sphere.R         == pytest.approx(1.910259417)
    # 1.910259417 is the numerical value when first implemented (and tested) 
    # and is specified here as a regression test.


def test_kernel_density_estimate_constructor():
    with pytest.raises(ValueError):
        sphere = md.KernelDensityEstimate(2, [np.array([0.5,1.5])], hyper_parameters=[0.1])
    with pytest.raises(ValueError):
        sphere = md.HyperSphere(2, [])
    with pytest.raises(ValueError):
        sphere = md.HyperSphere(0, [], hyper_parameters=[0.1])

    ndim    = 3
    density = md.KernelDensityEstimate(ndim, [], hyper_parameters=[0.1])

    assert density.ndim                == ndim
    assert density.D                   == pytest.approx(0.1)
    assert density.scales_set          == False
    assert density.start_end.shape[0]  == ndim
    assert density.start_end.shape[1]  == 2
    assert density.inv_scales.shape[0] == ndim
    assert density.radius_squared            == 0.05**2
    assert density.ngrid               == 13
    assert density.ln_norm             == pytest.approx(0.0)
    assert density.fitted              == False

    for i_dim in range(ndim):
        assert density.inv_scales[i_dim] == pytest.approx(1.0)
        for i_row in range(2):
            assert density.start_end[i_dim,i_row]   == pytest.approx(0.0)

    assert len(density.grid)  == 0

    return

def test_set_scales():

    ndim = 2
    domain = []
    hyper_parameters = [0.1]

    density = md.KernelDensityEstimate(ndim, domain, hyper_parameters=hyper_parameters)

    nsamples = 10
    with pytest.raises(ValueError):
        density.fit(np.ones((nsamples,ndim+1)),np.ones(nsamples))

    X      = np.zeros((2,ndim))-3.0
    X[1,:] = 2.0

    density.set_scales(X)

    for i_dim in range(ndim):
        assert density.inv_scales[i_dim]          == pytest.approx(1.0/5.0)
        assert density.inv_scales_squared[i_dim]  == pytest.approx(1.0/25.0)
        assert density.start_end[i_dim,0]         == pytest.approx(-3.0)
        assert density.start_end[i_dim,1]         == pytest.approx(2.0)

def test_kernel_density_estimate_precompute_normalising_factor():

    ndim = 2
    domain = []
    hyper_parameters = [0.1]

    density = md.KernelDensityEstimate(ndim, domain, hyper_parameters=hyper_parameters)

    nsamples = 10
    with pytest.raises(ValueError):
        density.fit(np.ones((nsamples,ndim+1)),np.ones(nsamples))

    X = np.zeros((3,ndim))
    X[0,:] = 0.0
    X[1,:] = 10.0
    X[2,:] = 5.0

    density.set_scales(X)
    density.precompute_normalising_factor(X)

    assert density.D        == pytest.approx(0.1)
    assert density.radius_squared == pytest.approx(0.0025)
    assert density.ln_norm  == pytest.approx(np.log(.25*np.pi)+np.log(3))

    X = np.zeros((4,ndim))
    X[0,:] = -10.0
    X[1,:] = 10.0
    X[2,:] = 5.0

    density.set_scales(X)
    density.precompute_normalising_factor(X)

    assert density.D        == pytest.approx(0.1)
    assert density.radius_squared == pytest.approx(0.0025)
    assert density.ln_norm  == pytest.approx(np.log(1.*np.pi)+np.log(4))

    ndim   = 6
    domain = []
    hyper_parameters = [0.2]
    density = md.KernelDensityEstimate(ndim, domain, hyper_parameters=hyper_parameters)

    X = np.zeros((3,ndim))
    X[0,:] = -10.0
    X[1,:] = 1.0
    X[2,:] = 10.0

    density.set_scales(X)
    density.precompute_normalising_factor(X)

    assert density.D          == pytest.approx(0.2)
    assert density.radius_squared   == pytest.approx(0.01)
    assert density.ln_norm    == pytest.approx(5.801314+np.log(3)) 
    # 5.801314 taken from wolfram alpha log(volume) of 6D r=2 sphere

    return

def test_kernel_density_estimate_fit():

    ndim = 2
    domain = []
    hyper_parameters = [0.1]

    density = md.KernelDensityEstimate(ndim, domain, hyper_parameters=hyper_parameters)

    nsamples = 1000

    with pytest.raises(ValueError):
        density.fit(np.ones((nsamples,ndim+1)),np.ones(nsamples))
    with pytest.raises(ValueError):
        density.fit(np.ones((nsamples,ndim)),np.ones(nsamples+1))

    X = np.zeros((3,ndim))
    X[0,:] = 0.01
    X[1,:] = 10.01
    X[2,:] = 5.01
    Y = -np.sum(X*X,axis=1)/2.0

    assert density.fit(X, Y)    == True
    print(density.grid, density.ngrid)
    assert density.grid[14][0]       == 0
    assert density.grid[11*13+11][0] == 1
    assert density.grid[6*13+6][0]   == 2
    assert density.fitted            == True

    return

def test_kernel_density_estimate_predict():

    ndim = 2
    domain = []
    hyper_parameters = [0.1]

    density = md.KernelDensityEstimate(ndim, domain, hyper_parameters=hyper_parameters)

    nsamples = 4

    X = np.zeros((4,ndim))
    X[0,:] = 0.00
    X[1,:] = 10.00
    X[2,:] = 5.00
    X[3,0] = 5.01
    X[3,1] = 5.00
    Y = -np.sum(X*X,axis=1)/2.0

    density.fit(X, Y)
    assert density.predict(np.array([0.0,0.0]))    == pytest.approx(np.log(1)-np.log(.25*np.pi)-np.log(nsamples))
    assert density.predict(np.array([0.0,0.501]))  == -np.inf
    assert density.predict(np.array([5.0,5.0]))    == pytest.approx(np.log(2)-np.log(.25*np.pi)-np.log(nsamples))
    assert density.predict(np.array([5.5005,5.0])) == pytest.approx(np.log(1)-np.log(.25*np.pi)-np.log(nsamples))

    # test if normalised
    n_grid = 100
    grid_start  = -2.0
    grid_length = 14.0
    post_grid = np.zeros((n_grid,n_grid))
    for i_x in range(n_grid):
        for i_y in range(n_grid):
            x = grid_start +  grid_length*i_x/n_grid
            y = grid_start +  grid_length*i_y/n_grid
            post_grid[i_x,i_y] = np.exp(density.predict(np.array([x,y])))

    assert np.sum(post_grid)*(grid_length/n_grid)*(grid_length/n_grid) == pytest.approx(1.0,rel=1E-2)

    # plt.imshow(post_grid)
    # plt.show()

    return

def test_beta_to_weights():
    
    ndim = 3

    beta   = np.zeros(ndim)
    weights = md.beta_to_weights_wrap(beta, ndim)
    

    for i_dim in range(ndim):
        assert weights[i_dim] == pytest.approx(1/3.0)

    beta[1] = 1.0
    weights = md.beta_to_weights_wrap(beta, ndim)
    norm = (2+np.exp(1))
    assert weights[0] == pytest.approx(1/norm)
    assert weights[1] == pytest.approx(np.exp(1.0)/norm)
    assert weights[2] == pytest.approx(1/norm)

    return

def test_calculate_gaussian_normalisation():

    np.random.seed(0)

    alpha = 1.5
    ndim  = 10

    cov = np.zeros((ndim,ndim))
    diag_cov = np.ones(ndim) + np.random.randn(ndim)*0.1
    np.fill_diagonal(cov, diag_cov*alpha)

    norm = md.calculate_gaussian_normalisation_wrap(alpha, 1.0/diag_cov, ndim)

    assert norm == pytest.approx(1.0/np.sqrt(np.linalg.det(cov*2*np.pi)))

    return

def test_evaluate_one_guassian():

    np.random.seed(0)

    ntrials = 20
    ndim    = 5

    alpha = 3.0
    mu = np.random.randn(ndim)
    diag_cov = np.ones(ndim) + np.random.randn(ndim)*0.1
    inv_covariance = 1.0/diag_cov
    weight = 0.2

    for i_trials in range(ntrials):
        x = np.random.randn(ndim)*5*i_trials/ntrials
        y = md.evaluate_one_guassian_wrap(x, mu, inv_covariance, alpha, weight, ndim)
        norm = md.calculate_gaussian_normalisation_wrap(alpha, inv_covariance, ndim)
        # assert y == -np.sum((x-mu)*(x-mu)*inv_covariance)
        assert y == np.exp(-np.sum((x-mu)*(x-mu)*inv_covariance)/(2.0*alpha))*norm*weight

    return

# def test_delta_theta_ij():

#     np.random.seed(0)

#     ntrials = 20
#     ndim    = 5

#     alpha = 3.0
#     mu = np.random.randn(ndim)
#     diag_cov = np.ones(ndim) + np.random.randn(ndim)*0.1
#     inv_covariance = 1.0/diag_cov

#     for i_trials in range(ntrials):
#         x = np.random.randn(ndim)*5*i_trials/ntrials
#         y = md.delta_theta_ij_wrap(x, mu, inv_covariance, ndim)
#         assert y == np.sum((x-mu)*(x-mu)*inv_covariance)

#     return

def test_ModifiedGaussianMixtureModel_constructor():

    ndim       = 4
    domains    = [np.array([1E-1,1E1])]
    gamma      = 1E-8
    nguassians = 3

    with pytest.raises(ValueError):
        MGMM = md.ModifiedGaussianMixtureModel(ndim, [np.array([1E-1,1E1])], hyper_parameters=[5])
    with pytest.raises(ValueError):
        MGMM = md.ModifiedGaussianMixtureModel(ndim, [np.array([1E-1,1E1]),np.array([1E-1,1E1])], hyper_parameters=[4,2,None,None,None])
    with pytest.raises(ValueError):
        MGMM = md.ModifiedGaussianMixtureModel(0, [np.array([0.5,1.5])], hyper_parameters=[5,1E-1,None,None,None])

    MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,None,None,None])

    assert MGMM.ndim            == ndim
    assert MGMM.alpha_domain[0] == domains[0][0]
    assert MGMM.alpha_domain[1] == domains[0][1]
    assert MGMM.nguassians      == nguassians
    for i_guas in range(nguassians):
        assert MGMM.beta_weights[i_guas] == pytest.approx(0.0)
        assert MGMM.alphas[i_guas]        == pytest.approx(1.0)
        for i_dim in range(ndim):
            assert MGMM.centres[i_guas,i_dim]         == pytest.approx(0.0)
            assert MGMM.inv_covariance[i_guas,i_dim]  == pytest.approx(1.0)
    assert MGMM.learning_rate == 0.1
    assert MGMM.centres_inv_cov_set == False

    return    

def test_ModifiedGaussianMixtureModel_set_weights():

    ndim       = 4
    domains    = [np.array([1E-1,1E1])]
    nguassians = 3
    gamma      = 1E-8

    MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,None,None,None])

    with pytest.raises(ValueError):
        MGMM.set_weights(np.ones(nguassians+1))
    with pytest.raises(ValueError):
        tmp_weights    = np.ones(nguassians)
        tmp_weights[1] = np.nan
        MGMM.set_weights(tmp_weights)
    with pytest.raises(ValueError):
        tmp_weights    = np.ones(nguassians)
        tmp_weights[1] = -1.0
        MGMM.set_weights(tmp_weights)
    with pytest.raises(ValueError):
        MGMM.set_weights(np.zeros(nguassians))

    MGMM.set_weights(np.ones(nguassians))

    for i_dim in range(nguassians):
        assert MGMM.beta_weights[i_dim] == pytest.approx(0.0)

    weights    = np.zeros(nguassians)
    weights[1] = 1.0
    MGMM.set_weights(weights)

    assert MGMM.beta_weights[0] == -np.inf
    assert MGMM.beta_weights[1] == pytest.approx(0.0)
    assert MGMM.beta_weights[2] == -np.inf

    return


def test_ModifiedGaussianMixtureModel_set_alphas():

    ndim       = 4
    domains    = [np.array([1E-1,1E1])]
    nguassians = 3
    gamma      = 1E-8

    MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,None,None,None])

    with pytest.raises(ValueError):
        MGMM.set_alphas(np.ones(nguassians+1))
    with pytest.raises(ValueError):
        tmp_alphas    = np.ones(nguassians)
        tmp_alphas[1] = np.nan
        MGMM.set_alphas(tmp_alphas)
    with pytest.raises(ValueError):
        tmp_alphas    = np.ones(nguassians)
        tmp_alphas[1] = -1.0
        MGMM.set_alphas(tmp_alphas)

    np.random.seed(0)
    random_array = np.random.uniform(size=nguassians)
    MGMM.set_alphas(random_array)

    for i_guas in range(nguassians):
        assert MGMM.alphas[i_guas] == pytest.approx(random_array[i_guas])

    return

def test_ModifiedGaussianMixtureModel_set_centres():

    ndim       = 4
    domains    = [np.array([1E-1,1E1])]
    nguassians = 3
    gamma      = 1E-8

    MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,None,None,None])

    with pytest.raises(ValueError):
        MGMM.set_centres(np.ones((nguassians+1,ndim)))
    with pytest.raises(ValueError):
        MGMM.set_centres(np.ones((nguassians,ndim+1)))
    with pytest.raises(ValueError):
        tmp_centres      = np.ones((nguassians,ndim))
        tmp_centres[1,2] = np.nan
        MGMM.set_centres(tmp_centres)

    np.random.seed(0)
    random_array = np.random.randn(nguassians,ndim)
    MGMM.set_centres(random_array)

    for i_guas in range(nguassians):
        for i_dim in range(ndim):
            assert MGMM.centres[i_guas,i_dim] == pytest.approx(random_array[i_guas,i_dim])

    return

def test_ModifiedGaussianMixtureModel_set_inv_covarience():

    ndim       = 4
    domains    = [np.array([1E-1,1E1])]
    nguassians = 3
    gamma      = 1E-8

    MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,None,None,None])

    with pytest.raises(ValueError):
        MGMM.set_inv_covariance(np.ones((nguassians+1,ndim)))
    with pytest.raises(ValueError):
        MGMM.set_inv_covariance(np.ones((nguassians,ndim+1)))
    with pytest.raises(ValueError):
        tmp_inv_covariences      = np.ones((nguassians,ndim))
        tmp_inv_covariences[1,2] = np.nan
        MGMM.set_inv_covariance(tmp_inv_covariences)
    with pytest.raises(ValueError):
        tmp_inv_covariences      = np.ones((nguassians,ndim))
        tmp_inv_covariences[1,2] = -1.0
        MGMM.set_inv_covariance(tmp_inv_covariences)

    np.random.seed(0)
    random_array = 1.0 + np.random.randn(nguassians,ndim)*0.1
    MGMM.set_inv_covariance(random_array)

    for i_guas in range(nguassians):
        for i_dim in range(ndim):
            assert MGMM.inv_covariance[i_guas,i_dim] == pytest.approx(random_array[i_guas,i_dim])

    return

def test_ModifiedGaussianMixtureModel_set_inv_covarience():

    ndim       = 4
    domains    = [np.array([1E-1,1E1])]
    nguassians = 3
    gamma      = 1E-8

    MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,None,None,None])

    with pytest.raises(ValueError):
        MGMM.set_centres_and_inv_covariance(np.ones((nguassians+1,ndim)),np.ones((nguassians,ndim)))
    with pytest.raises(ValueError):
        MGMM.set_centres_and_inv_covariance(np.ones((nguassians,ndim+1)),np.ones((nguassians,ndim)))
    with pytest.raises(ValueError):
        tmp_inv_covariences      = np.ones((nguassians,ndim))
        tmp_inv_covariences[1,2] = np.nan
        MGMM.set_centres_and_inv_covariance(tmp_inv_covariences,np.ones((nguassians,ndim)))
    with pytest.raises(ValueError):
        MGMM.set_centres_and_inv_covariance(np.ones((nguassians,ndim)),np.ones((nguassians+1,ndim)))
    with pytest.raises(ValueError):
        MGMM.set_centres_and_inv_covariance(np.ones((nguassians,ndim)),np.ones((nguassians,ndim+1)))
    with pytest.raises(ValueError):
        tmp_inv_covariences      = np.ones((nguassians,ndim))
        tmp_inv_covariences[1,2] = np.nan
        MGMM.set_centres_and_inv_covariance(np.ones((nguassians,ndim)),tmp_inv_covariences)
    with pytest.raises(ValueError):
        tmp_inv_covariences      = np.ones((nguassians,ndim))
        tmp_inv_covariences[1,2] = -1.0
        MGMM.set_centres_and_inv_covariance(np.ones((nguassians,ndim)),tmp_inv_covariences)

    np.random.seed(0)
    random_array1 = np.random.randn(nguassians,ndim)
    random_array2 = 1.0 + np.random.randn(nguassians,ndim)*0.1
    MGMM.set_centres_and_inv_covariance(random_array1, random_array2)

    for i_guas in range(nguassians):
        for i_dim in range(ndim):
            assert MGMM.centres[i_guas,i_dim]        == pytest.approx(random_array1[i_guas,i_dim])
            assert MGMM.inv_covariance[i_guas,i_dim] == pytest.approx(random_array2[i_guas,i_dim])

    assert MGMM.centres_inv_cov_set == True

    return
def test_ModifiedGaussianMixtureModel_predict():

    np.random.seed(0)

    ntrials    = 20
    ndim       = 50
    nguassians = 25
    gamma      = 1E-8
    domains    = [np.array([1E-1,1E1])]

    alphas         = np.abs(np.ones(nguassians) + np.random.randn(nguassians)*0.2)
    mus            = np.random.randn(nguassians,ndim)
    diag_cov       = np.abs(np.ones((nguassians,ndim)) + np.random.randn(nguassians,ndim)*0.2)
    inv_covariance = 1.0/diag_cov
    weights        = np.random.uniform(size=(nguassians))
    weights        = weights/np.sum(weights)

    MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,None,None,None])

    MGMM.set_alphas(alphas)
    MGMM.set_centres(mus)
    MGMM.set_inv_covariance(inv_covariance)
    MGMM.set_weights(weights)

    for i_trials in range(ntrials):
        x = np.random.randn(ndim)*5*i_trials/ntrials
        y = MGMM.predict(x)
        y_compare = 0.0
        for i_guas in range(nguassians):
            norm = md.calculate_gaussian_normalisation_wrap(alphas[i_guas], inv_covariance[i_guas,:], ndim)
            y_compare += np.exp(-np.sum((x-mus[i_guas,:])*(x-mus[i_guas,:])*inv_covariance[i_guas,:])/(2.0*alphas[i_guas]))*norm*weights[i_guas]
        assert y == pytest.approx(np.log(y_compare))

    return


def test_ModifiedGaussianMixtureModel_fit():

    np.random.seed(2)

    nsamples   = 2000
    ndim       = 2
    nguassians = 2
    gamma      = 1E-30
    domains    = [np.array([1E-2,10E0])]
    sigma1     = 2.0
    sigma2     = 4.0
    mu_off     = 20.0

    MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,1.0,20,10])
    # MGMM.verbose = True

    X = np.zeros((nsamples,ndim))
    Y = np.zeros((nsamples))
    X[:nsamples//4,:] = np.random.randn(nsamples//4,ndim)*sigma1
    X[nsamples//4:,:] = np.random.randn(3*nsamples//4,ndim)*sigma2 + mu_off
    Y[:nsamples//4] = -np.sum(X[:nsamples//4,:]*X[:nsamples//4,:]/(2.0*sigma1*sigma1),axis=1)
    Y[nsamples//4:] = -np.sum((X[nsamples//4:,:]-mu_off)**2/(2.0*sigma2*sigma2),axis=1)

    MGMM.fit(X, Y)

    centre_1 = (MGMM.centres[0,0], MGMM.centres[0,1])
    centre_2 = (MGMM.centres[1,0], MGMM.centres[1,1])
    centre_lo, centre_hi = (centre_1, centre_2) if (centre_1[0] < centre_2[0]) \
        else (centre_2, centre_1)
    assert centre_hi[0] == pytest.approx(20.19605982) # makes sense as close to 20.0
    assert centre_hi[1] == pytest.approx(19.69715662)
    assert centre_lo[0] == pytest.approx(-0.02629883) # makes sense as close to 0.0
    assert centre_lo[1] == pytest.approx(-0.16510091)

    inv_cov_1 = (MGMM.inv_covariance[0,0], MGMM.inv_covariance[0,1])
    inv_cov_2 = (MGMM.inv_covariance[1,0], MGMM.inv_covariance[1,1])
    inv_cov_lo, inv_cov_hi = (inv_cov_1, inv_cov_2) if (inv_cov_1[0] < inv_cov_2[0]) \
        else (inv_cov_2, inv_cov_1)
    assert inv_cov_lo[0] == pytest.approx(0.06037615) # makes sense as close to 1/(4.0**2)
    assert inv_cov_lo[1] == pytest.approx(0.06203164)
    assert inv_cov_hi[0] == pytest.approx(0.24818792) # makes sense as close to 1/(2.0**2)
    assert inv_cov_hi[1] == pytest.approx(0.24781514)

    alpha_1 = MGMM.alphas[0]
    alpha_2 = MGMM.alphas[1]
    alpha_lo, alpha_hi = (alpha_1, alpha_2) if (alpha_1 < alpha_2) \
        else (alpha_2, alpha_1)
    assert alpha_lo == pytest.approx(0.97125603, abs=1e-3) # makes sense as close to 1
    assert alpha_hi == pytest.approx(0.99537613, abs=1e-3) # makes sense as close to 1

    norm    = np.sum(np.exp(MGMM.beta_weights))
    weights = np.exp(MGMM.beta_weights)/norm

    weight_1 = weights[0]
    weight_2 = weights[1]
    weight_lo, weight_hi = (weight_1, weight_2) if (weight_1 < weight_2) \
        else (weight_2, weight_1)

    assert weight_lo == pytest.approx(0.25252212) # makes sense as close to 0.25
    assert weight_hi == pytest.approx(0.74747788) # makes sense as close to 0.75


    return

# def test_idea(sigma):

#     np.random.seed(0)

#     ntrials    = 100
#     nsamples   = 2000
#     ndim       = 4
#     nguassians = 1
#     gamma      = 1E-8
#     domains    = [np.array([1E-20,5E0])]

#     mus            = np.zeros((nguassians,ndim))
#     diag_cov       = np.ones((nguassians,ndim))*sigma*sigma
#     inv_covariance = 1.0/diag_cov
#     weights        = np.ones((nguassians))

#     MGMM = md.ModifiedGaussianMixtureModel(ndim, domains, hyper_parameters=[nguassians, gamma,None,None,None])

#     MGMM.set_centres(mus)
#     MGMM.set_inv_covariance(inv_covariance)
#     MGMM.set_weights(weights)

#     O = np.zeros(ntrials)

#     x = np.random.randn(nsamples,ndim)*sigma
#     for i_trials in range(ntrials):
#         alphas    = np.array([domains[0][0] + (domains[0][1]-domains[0][0])*i_trials/ntrials])
#         MGMM.set_alphas(alphas)

#         for i_sample in range(nsamples):
#             y = MGMM.predict(x[i_sample,:])
#             O[i_trials] += np.exp(2*y+np.dot(x[i_sample,:],x[i_sample,:])/(sigma**2)-100)


#     plt.plot(np.linspace(0,domains[0][1],ntrials),O)
#     plt.show()

#     return

# # test_idea(sigma)


