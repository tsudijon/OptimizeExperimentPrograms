import numpy as np
from scipy import stats
from scipy.optimize import minimize

def evaluate_production_function(n, prior, sigma, num_mc = 100_000):
    """
    n: input to the production_function
    prior: discrete probability measure consisting of tuple (atoms, weights)
        where atoms is a numpy array of atom locations and weights are a numpy array of weights, summing to one.
    sigma: population std parameter.
    """
    atoms, weights = prior
    weights = weights/weights.sum()

    mc_samples = np.random.choice(atoms, size=(num_mc,1), p= weights) + np.random.normal(0,sigma/np.sqrt(n),size = (num_mc,1))

    # calculate Delta_i - hat{Delta}
    data = atoms.T - mc_samples  
    data = stats.norm.pdf(data*(np.sqrt(n)/sigma)) * weights
    post_weights = data/ data.sum(axis = 1).reshape(-1,1)
    return np.maximum(post_weights @ atoms,0).mean()


######### Gaussian Case ###############

def gaussian_production_function(n, tau, mu, sigma):
  def mvar(n):
    return tau**2 + sigma**2/n

  return tau**2/(np.sqrt(2*np.pi*mvar(n))) * np.exp(-mu**2 * mvar(n)/(2* tau**4)) + mu*stats.norm.cdf(mu*np.sqrt(mvar(n))/tau**2) - np.max(mu,0)