"""This module defines the GARCH model object, allowing for exogenous variables in the volatility component."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.optimize as opt


class GARCH:
    """
    This class defines the GARCH model object.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.9, omega: float = 0.1,
                 gammas: float = 0.1, mu=None, p: int = 1, q: int = 1, z: int = 1):
        """
        Initialize GARCH model parameters.

        Parameters
        ----------
        alpha : float, optional
            The ARCH component parameter, by default 0.1.
        beta : float, optional
            The GARCH component parameter, by default 0.9.
        omega : float, optional
            The constant term in the variance equation, by default 0.1.
        gammas : float, optional
            The coefficient for exogenous variables, by default 0.1.
        mu : float, optional
            Mean of the returns, by default None.
        p : int, optional
            Lag order for the ARCH component, by default 1.
        q : int, optional
            Lag order for the GARCH component, by default 1.
        z : int, optional
            Lags in exogenous variable, by default 1. If 
        """
        self.alpha = np.array([alpha] * p)
        self.beta = np.array([beta] * q)
        self.omega = omega
        self.gammas = gammas  # Used when exogenous variables are present
        self.p = p
        self.q = q
        self.z = z
        self.mu = 0 if mu is None else mu
        self.sigma2 = np.array([])
        self.y = np.array([])


    def __repr__(self):
        return f"omega = {self.omega:.3g}\nalpha = {self.alpha}\nbeta = {self.beta}"


    def train(self, y: pd.Series, x=None, callback_func=None):
        """
        Estimate parameters using maximum likelihood and fit the GARCH model to the data.

        Parameters
        ----------
        y : pd.Series
            Time series data (e.g., returns).
        x : np.array, optional
            Exogenous variables for the model, by default None.
        callback_func : function, optional
            A callback function for monitoring optimization, by default None.
        """
        self.mu = np.mean(y)
        self.y = np.array(y)
        self.n_obs = len(y)
        e_t = self.y - self.mu
        self.e_t = e_t  # Clean this part up.
        self.x = x

        try:
            self.y_index = y.index
        except Exception as e:
            self.y_index = np.arange(self.n_obs)

        init_omega = self.omega
        init_alpha = self.alpha
        init_beta = self.beta

        if x is not None:
            x = np.array(x)
            exo_var_count = x.shape[1]
            init_gammas = np.array([self.gammas] * exo_var_count * self.z)
            init_params = self.inv_repam([init_omega, *init_alpha, *init_beta, *init_gammas])
        else:
            init_params = self.inv_repam([init_omega, *init_alpha, *init_beta])
       
        print('optimising')

        opt_result = opt.minimize(
            self.log_likelihood,
            x0=init_params,
            args=(y, e_t, x, True),
            method='BFGS',
            callback=callback_func,
            options={'maxiter': 100}
        )
        self.opt_result = opt_result
        print('optimising finished')
       
       # Save estimated parameters after fitting.
        omega = self.repam(opt_result.x[0])
        alpha = self.repam(opt_result.x[1: self.p + 1])
        beta = self.repam(opt_result.x[self.p + 1: self.p + self.q + 1])

        if x is not None:
            gammas = self.repam(opt_result.x[self.p + self.q + 1 :]).reshape(self.z, x.shape[1])
            print(f"{omega=}\n {alpha=}\n {beta=}\n {gammas=}")
        else:
            print(f"{omega=}\n {alpha=}\n {beta=}\n")

        # Compute sigma2 values using optimised parameters.
        self.sigma2 = np.zeros(self.n_obs)
        self.sigma2[0] = np.var(y)
        
        for t in range(max(self.p, self.q, self.z), self.n_obs):
            if x is not None:
                #print(x[t - self.z + 1 : t + 1, :])
                self.sigma2[t] = (omega + np.sum(alpha * (e_t[t - self.p : t] ** 2)) 
                             + np.sum(beta * (self.sigma2[t - self.q : t])) 
                             + np.sum(gammas * x[t - self.z + 1 : t + 1, :])) 
            else:
                self.sigma2[t] = omega + np.sum(alpha * (e_t[t - self.p: t] ** 2)) + np.sum(beta * (self.sigma2[t - self.q: t]))

        #print('\nResults of BFGS minimization\n{}\n{}'.format(''.join(['-']*28), opt_result))
        #print('\nResulting params = {}'.format(self.params))


    def calculate_score(self):
        # omega
        del_tminus1 = 0
        score_omega = [del_tminus1]  # Store the partial derivative over time

        for n in range(self.n_obs): 
            del_tminus1 = 1 + self.beta[0] * del_tminus1
            score_omega.append(del_tminus1)

        # alpha
        del_tminus1 = 0
        score_alpha = [del_tminus1]  # Store the partial derivative over time
        for n in range(self.n_obs): 
            del_tminus1 = (self.y[n] - self.mu)**2 + self.beta[0] * del_tminus1
            score_alpha.append(del_tminus1)

 
        del_tminus1 = 0
        score_beta = [del_tminus1]  # Store the partial derivative over time
        for n in range(self.n_obs): 
            del_tminus1 = self.sigma2[n] + self.beta[0] * del_tminus1
            score_beta.append(del_tminus1)

        omega = np.sum(np.array(score_omega) ** 2) / self.n_obs  

        return score_omega, score_alpha, score_beta


    def log_likelihood(self, params_repam, y: pd.Series, e_t, x=None, fmin=False):
        """
        Calculate the log likelihood of the GARCH model.

        Parameters
        ----------
        params_repam : np.array
            Reparameterized parameter array.
        y : pd.Series
            Time series data.
        e_t : np.array
            Residuals from the model mean.
        x : np.array, optional
            Exogenous variables, by default None.
        fmin : bool, optional
            If True, return only the likelihood value, by default False.
        """
        p, q, z = self.p, self.q, self.z
        omega, alpha, beta, gammas = self._parse_params(params_repam, x)
        
        # Calculation for log likelihood
        t_max = len(y)
        avg_log_like = 0
        sigma2 = np.zeros(t_max)
        sigma2[:max(p, q, z)] = np.var(y)

        for t in range(max(p, q, z), t_max):
            if x is not None:
                sigma2[t] = (omega + np.sum(alpha * (e_t[t - p : t] ** 2)) 
                             + np.sum(beta * (sigma2[t - q : t])) 
                             + np.sum(gammas * x[t - z + 1 : t + 1, :] ** 2)) 
            else:
                sigma2[t] = omega + np.sum(alpha * (e_t[t - p: t] ** 2)) + np.sum(beta * (sigma2[t - q: t]))

            avg_log_like += (np.log(sigma2[t]) + (y[t] - self.mu) ** 2 / sigma2[t]) / t_max

        return avg_log_like if fmin else [avg_log_like, sigma2]
    

    def deviance(self, theta) -> float:
        """Theta is an array representing the paramters."""
        # log likelihood of our estimated
        print(f"{self.opt_result.x}")

        assert len(theta) == len(self.opt_result.x) 

        if self.x is not None:
            l_p_hat = self.log_likelihood(self.opt_result.x, y=self.y, e_t=self.e_t, x=self.x, fmin=True)
            l_p_theta = self.log_likelihood(theta, y=self.y, e_t=self.e_t, x=self.x, fmin=True)
        else:
            l_p_hat = self.log_likelihood(self.opt_result.x, y=self.y, e_t=self.e_t, fmin=True)
            l_p_theta = self.log_likelihood(theta, y=self.y, e_t=self.e_t, fmin=True)
        
        res = 2 * (l_p_hat - l_p_theta)
        return res

    def _parse_params(self, params_repam, x):
        """Helper function to parse reparameterized parameters into usable form."""
        omega = self.repam(params_repam[0])
        alpha = self.repam(params_repam[1: self.p + 1])
        beta = self.repam(params_repam[self.p + 1: self.p + self.q + 1])
        gammas = None if x is None else self.repam(params_repam[self.p + self.q + 1:]).reshape(self.z, x.shape[1])
        return omega, alpha, beta, gammas


    def plot(self, figsize=(11, 5), title=None, ts_label='Time Series', vol_label='Conditional Volatility'):
        """
        Plot the log returns and conditional volatility.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (11, 5).
        """
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.plot(self.y_index, self.y, label='Log Returns', color='blue', alpha=0.5)
        ax1.set_ylabel(ts_label, color='blue')

        ax2 = ax1.twinx()
        ax2.plot(self.y_index, self.sigma2, label='Conditional Volatility', color='orange')
        ax2.set_ylabel(vol_label, color='orange')

        if not title:
            title = f'GARCH(p={self.p}, q={self.q}, z={self.z})'

        plt.title(title)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.grid()
        plt.show()


    def repam(self, params):
        """Reparameterize parameters for optimization stability."""
        return np.exp(params)

    def inv_repam(self, params):
        """Inverse reparameterization for optimization stability."""
        return np.log(params)



if __name__ == "__main__":
    import random
    # Define volatility component coefficients
    omega, alpha, beta, gamma = 0.1, 0.1, 0.4, 0.05
    T = 200

    e = [0]  # errors e_t
    sigma2 = [1] # sigma sigma_t
    x = np.random.randn(T, 1)

    for t in range(T):
        sigma2_t = omega + alpha * e[-1]**2 + beta * sigma2[-1] + gamma * x[t]**2
        e_t = random.gauss(0, sigma2_t**0.5) 

        e.append(e_t)
        sigma2.append(sigma2_t)

    e = e[1:]
    sigma2 = sigma2[1:]


    garch_with_exo = GARCH(p=1, q=1, z=0)
    garch_with_exo.train(e)

    betas = np.arange(0, 0.8, 0.05)
    print(betas)
    opt_result = garch_with_exo.opt_result.x

    deviance_array = []

    for i in betas:
        
        theta = [garch_with_exo.opt_result.x[0], i, garch_with_exo.opt_result.x[2:]]
        print(theta)
        beta_deviance = garch_with_exo.deviance(theta)
        deviance_array.append(beta_deviance)
    
    #garch_without_exo = GARCH(p=1, q=1, z=0)
    #garch_without_exo.train(e)
    print(deviance_array)