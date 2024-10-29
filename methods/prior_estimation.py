import numpy as np
from scipy import stats
from scipy.optimize import minimize
from npeb import GLMixture

def empirical_bayes_normal(estimates, standard_errors):
    """
    Normal ebayes model

    Parameters:
    estimates : Estimated treatment effects
    standard_errors : Standard errors of the estimates
    """
    # Initial estimates for mu and tau
    mu_init, tau_init = np.mean(estimates), np.std(estimates)
    # negative log-likelihood function
    def neg_log_likelihood(params):
        mu, tau = params
        return -np.sum(
                stats.norm.logpdf(estimates,
                    mu,
                    np.sqrt(standard_errors**2 + tau**2)
                ))
    result = minimize(
        neg_log_likelihood,
        [mu_init, tau_init],
        method='L-BFGS-B',
        bounds=[(None, None), 
                (0, None),
               ]
    )
    mu, tau = result.x
    # Calculate the Empirical Bayes estimates
    alpha = tau**2 / (tau**2 + standard_errors**2)
    eb_estimates = alpha * estimates + (1 - alpha) * mu
    return eb_estimates, mu, tau