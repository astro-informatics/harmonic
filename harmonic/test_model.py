import pytest
import model as md
import numpy as np

def test_hyper_sphere_constructor():

    with pytest.raises(ValueError):
        sphere = md.HyperSphere(2, [np.array([0.5,1.5])], hyper_parameters=[5])
    with pytest.raises(ValueError):
        sphere = md.HyperSphere(2, [np.array([0.5,1.5]),np.array([0.5,1.5])])

    ndim   = 3
    domain = [np.array([0.5,1.5])]
    sphere = md.HyperSphere(ndim, domain)

    assert sphere.ndim               == ndim
    assert sphere.R_domain.shape[0]  == 2
    assert sphere.R_domain[0]        == pytest.approx(0.5)
    assert sphere.R_domain[1]        == pytest.approx(1.5)
    assert sphere.R                  == pytest.approx(1.0)
    assert sphere.centre_set        == False
    assert sphere.inv_covarience_set == False

    for i_dim in range(ndim):
        assert sphere.centre[i_dim]        == pytest.approx(0.0)
        assert sphere.inv_covarience[i_dim] == pytest.approx(1.0)


def test_hyper_sphere_set_sphere_centre_and_shape():

    ndim   = 3
    domain = [np.array([0.5,1.5])]

    sphere = md.HyperSphere(ndim, domain)

    with pytest.raises(ValueError):
        sphere.set_centre(np.array([0.0,0.0]))
    with pytest.raises(ValueError):
        sphere.set_centre(np.array([0.0,0.0,np.nan]))

    with pytest.raises(ValueError):
        sphere.set_inv_covarience(np.array([0.0,0.0]))
    with pytest.raises(ValueError):
        sphere.set_inv_covarience(np.array([0.0,0.0,np.nan]))

    sphere.set_centre(np.array([1.5,3.0,2.0]))
    assert sphere.centre[0]  == pytest.approx(1.5)
    assert sphere.centre[1]  == pytest.approx(3.0)
    assert sphere.centre[2]  == pytest.approx(2.0)
    assert sphere.centre_set == True

    sphere.set_inv_covarience(np.array([2.5,4.0,-1.0]))
    assert sphere.inv_covarience[0]  == pytest.approx(2.5)
    assert sphere.inv_covarience[1]  == pytest.approx(4.0)
    assert sphere.inv_covarience[2]  == pytest.approx(-1.0)
    assert sphere.inv_covarience_set == True


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

    sphere.set_inv_covarience(np.full((ndim),2.0))

    assert sphere.ln_one_over_volume == pytest.approx(-5.801314-3*np.log(0.5))

def test_hyper_sphere_predict():

    ndim   = 6
    domain = [np.array([0.5,1.5])]
    sphere = md.HyperSphere(ndim, domain)
    
    assert sphere.predict(np.zeros((ndim))) == pytest.approx(-5.801314+6*np.log(2.0))

    sphere.set_R(4.0)

    assert sphere.predict(np.zeros((ndim)))                  == pytest.approx(-5.801314+6*np.log(0.5))
    assert sphere.predict(np.full((ndim),4.0))               == -np.inf
    assert np.exp(sphere.predict(np.full((ndim),4.0))+1E99)  == pytest.approx(0.0)

    x    = np.zeros((ndim))
    x[4] = 4.0001
    assert sphere.predict(x) == -np.inf
    x[4] = 3.9999
    assert sphere.predict(x) == pytest.approx(-5.801314+6*np.log(0.5))

    inv_covarience  = np.ones((ndim))*4
    sphere.set_inv_covarience(inv_covarience)
    x[4] = 2.0001
    assert sphere.predict(x) == -np.inf
    x[4] = 1.9999
    assert sphere.predict(x) == pytest.approx(-5.801314+6*np.log(0.5)+6*np.log(2.0))



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

    assert sphere.fit(X, Y) == True
    assert sphere.R         == pytest.approx(3.649091)

    np.random.seed(30)
    X = np.random.randn(nsamples,ndim)+1.0
    Y = -np.sum((X-1.0)**2,axis=1)/2.0

    assert sphere.fit(X, Y) == True
    assert sphere.R         == pytest.approx(3.649091)

    np.random.seed(30)
    X = np.random.randn(nsamples,ndim)
    X[:,1] = X[:,1]*0.5
    Y = -X[:,0]*X[:,0]/2.0 - X[:,1]*X[:,1]/(2.0*0.25)

    assert sphere.fit(X, Y) == True
    assert sphere.R         == pytest.approx(3.649091)

    return

def test_kernel_density_estimate_constructor():
    with pytest.raises(ValueError):
        sphere = md.KernelDensityEstimate(2, [np.array([0.5,1.5])], hyper_parameters=[5.])
    with pytest.raises(ValueError):
        sphere = md.HyperSphere(2, [])

    ndim    = 3
    density = md.KernelDensityEstimate(ndim, [], hyper_parameters=[0.1])

    assert density.ndim                == ndim
    assert density.R                   == pytest.approx(0.1)
    assert density.scales_set          == False
    assert density.start_end.shape[0]  == ndim
    assert density.start_end.shape[1]  == 2
    assert density.inv_scales.shape[0] == ndim
    assert density.distance            == 0.05**2
    assert density.ngrid               == 12

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

    X      = np.zeros((2,ndim))-3.0
    X[1,:] = 2.0

    density.set_scales(X)

    for i_dim in range(ndim):
        assert density.inv_scales[i_dim]  == 2.
        assert density.start_end[i_dim,0] == -3.0
        assert density.start_end[i_dim,1] == 2.0

def test_kernel_density_estimate_fit():

    ndim = 2
    domain = []
    hyper_parameters = [0.1]

    density = md.KernelDensityEstimate(ndim, domain, hyper_parameters=hyper_parameters)

    nsamples = 10000

    with pytest.raises(ValueError):
        density.fit(np.ones((nsamples,ndim+1)),np.ones(nsamples))
    with pytest.raises(ValueError):
        density.fit(np.ones((nsamples,ndim)),np.ones(nsamples+1))

    np.random.seed(30)
    X = np.random.randn(nsamples,ndim)
    Y = -np.sum(X*X,axis=1)/2.0

    assert density.fit(X, Y) == True
    # assert 1 in density.grid == False

    return