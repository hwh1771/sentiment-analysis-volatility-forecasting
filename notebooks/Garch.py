"""This module defines the GARCH model object, allowing for exogenous variables in the volatility component."""
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.optimize as opt
from scipy.stats import t 


class GARCH:
    """
    GARCH model with exogenous parameter in the volatility component.

    Parameters
    ----------
        alpha : float, optional
            The ARCH component parameter.
        beta : float, optional
            The GARCH component parameter.
        omega : float, optional
            The constant term in the variance equation.
        gamma : float, optional
            The coefficient for exogenous variables.
        mu : float, optional
            Mean of the returns, by default None.
        p : int, optional
            Lag order for the ARCH component, by default 1.
        q : int, optional
            Lag order for the GARCH component, by default 1.
        z : int, optional
            Lags in exogenous variable, by default 0.
        verbose: bool 
            Will print key steps during fitting.

    Example
    ----------
    from garch import GARCH
    >>> garch = GARCH(p=1, q=1, z=1, verbose=True)
    >>> garch.train(returns, x=x)

    >>> garch.summary()
        # Will print out parameter statistics

    >>> garch.loglikelihood

    """

    def __init__(self, alpha: float = 0, beta: float = 0, omega: float = 0,
                 gamma: float = None, mu=None, p: int = 1, q: int = 1, z: int = 0, verbose=False):
        """
        Initialize GARCH model parameters.

        Parameters
        ----------
        alpha : float, optional
            The ARCH component parameter.
        beta : float, optional
            The GARCH component parameter.
        omega : float, optional
            The constant term in the variance equation.
        gamma : float, optional
            The coefficient for exogenous variables.
        mu : float, optional
            Mean of the returns, by default None.
        p : int, optional
            Lag order for the ARCH component, by default 1.
        q : int, optional
            Lag order for the GARCH component, by default 1.
        z : int, optional
            Lags in exogenous variable, by default 0.
        verbose: bool 
            Will print key steps during fitting.
        """
        self.alpha = np.array([alpha] * p)
        self.beta = np.array([beta] * q)
        self.omega = omega
        self.gamma = gamma  # Used when exogenous variables are present
        self.model_params = {'omega': 0,
                            'alpha': [0],
                            'beta': [0],
                            'gamma': None}

        self.p = p
        self.q = q
        self.z = z
        self.mu = 0 if mu is None else mu
        self.sigma2 = np.array([])
        self.y = np.array([])
        self.verbose = verbose
        self.x = None


        def __repr__(self):
            return (f"GARCH(p={self.p}, q={self.q}, z={self.z}, omega={self.omega:.3g}, "
                    f"alpha={self.alpha.tolist()}, beta={self.beta.tolist()}, "
                    f"gamma={self.gamma if self.gamma is not None else 'None'})")


    def train(self, y: pd.Series, x=None, callback_func=None, maxiter=100,
              method='L-BFGS-B', suppress_warnings=False, use_constraints=False):
        """
        Estimate model parameters using MLE and fit the GARCH model to the data.

        Parameters
        ----------
        y : pd.Series
            Time series data (e.g., returns).
        x : np.array, optional
            Exogenous variables for the model. Shape is (T, k) for T time steps and 
            k exo variables. 
        callback_func : function, optional
            A callback function for monitoring optimization, by default None.
        """
        self.mu = np.mean(y)
        self.y = np.array(y)
        self.n_obs = len(y)
        e = self.y - self.mu

        if x is not None:
            assert len(x) == len(y), "Length of y and x are not equal."

            if isinstance(x, pd.DataFrame):
                self.x = x.to_numpy()
            elif isinstance(x, pd.Series):
                self.x = x.to_numpy().reshape(-1, 1)
            elif isinstance(x, np.ndarray) and x.ndim == 1:
                self.x = x.reshape(-1, 1)
            else:
                self.x = np.array(x)

        try:
            self.y_index = y.index
        except Exception:
            self.y_index = np.arange(self.n_obs)

        # Initialise parameter values.
        init_omega = 0.4
        init_alpha = np.array([0.3] * self.p)
        init_beta = np.array([0.3] * self.q)

        # Set bounds for optimisation, alpha and beta should be less than 1.
        opt_bounds = [(None, None)]
        opt_bounds.extend([(None, 0)]*self.p)
        opt_bounds.extend([(None, 0)]*self.q)

        if x is not None:
            exo_var_count = self.x.shape[1]

            init_gamma = np.array([0.5] * exo_var_count * self.z)
            init_params = self.inv_repam([init_omega, *init_alpha, *init_beta, *init_gamma])
            opt_bounds.extend([(None, None)]*self.z*exo_var_count)        
        else:
            init_params = self.inv_repam([init_omega, *init_alpha, *init_beta])

        if not suppress_warnings and self.calculate_log_likelihood(init_params, self.y, e, self.x, True) < 0:
            print('DataScaleWarning: y is poorly scaled, which may affect convergence of the optimizer when estimating the model parameters. \nRecommendation: pass in 100*y')

        if self.verbose:
            print('Optimising...')

        start = time.time()
        
        constraints=({'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2]}) if use_constraints else None
        opt_result = opt.minimize(
            self.calculate_log_likelihood,
            x0=init_params,
            args=(self.y, e, self.x, True),
            method=method,
            callback=callback_func,
            options={'maxiter': maxiter},
            bounds=opt_bounds,
            constraints = constraints
        )

        self.loglikelihood = -0.5*self.n_obs*(opt_result.fun + np.log(2*np.pi))
        opt_success: bool = opt_result.success
        

        end = time.time()      
        if self.verbose:
            print(f'Optimising finished in {(end-start):.3f}s')
            print(opt_result)
       
       # Save estimated parameters after fitting.
        self.model_params = self._parse_params(opt_result.x, self.x, as_dict=True)
        self.omega = self.model_params['omega']
        self.alpha = self.model_params['alpha']
        self.beta = self.model_params['beta']
        self.gamma = self.model_params['gamma'] if 'gamma' in self.model_params.keys() else None

        if self.verbose:
            print(self.model_params)

        # Compute sigma2(hat) values using optimised parameters.        
        self.sigma2 = self.compute_sigma2(e, init_sigma=np.var(y), x=self.x)

        self.information_matrix = self.calculate_information_matrix(e, self.sigma2, self.x)

        return opt_success


    def calculate_log_likelihood(self, params_repam, y: pd.Series, e, x=None, fmin=False):
        """
        Calculate the log likelihood of the GARCH model.

        Parameters
        ----------
        params_repam : np.array
            Reparameterized parameter array.
        y : pd.Series
            Time series data.
        e : np.array
            Residuals from the model mean.
        x : np.array, optional
            Exogenous variables, by default None.
        fmin : bool, optional
            If True, return only the likelihood value, by default False.
        """
        p, q, z = self.p, self.q, self.z
        omega, alpha, beta, gamma = self._parse_params(params_repam, x)

        #if alpha+beta >= 1:
        #    return 10
       
        # Calculation for log likelihood
        avg_log_like = 0
        sigma2 = np.zeros(self.n_obs)
        if x is not None:
            sigma2[:max(p, q, z)] =  np.var(y) #(omega+gamma*np.mean(x**2, axis=0)) / (1-alpha-beta)#np.var(y)
        else:
            sigma2[:max(p, q, z)] =  omega / (1-alpha-beta)

        for t in range(max(p, q, z), self.n_obs):
            if x is not None:
                sigma2[t] = (omega + np.sum(alpha * (e[t - p : t] ** 2))
                             + np.sum(beta * (sigma2[t - q : t])) 
                             + np.sum(gamma * (x[t - z + 1 : t + 1, :] ** 2)))
            else:
                sigma2[t] = omega + np.sum(alpha * (e[t - p: t] ** 2)) + np.sum(beta * (sigma2[t - q: t]))

            avg_log_like += (np.log(sigma2[t]) + (e[t] ** 2) / sigma2[t]) / self.n_obs 
        
        return avg_log_like if fmin else [avg_log_like, sigma2]


    def compute_sigma2_first_derivative(self, param: str, init_value=None, e=None, sigma2=None, x=None) -> list:
        """Compute first partial derivative of sigma squared to any parameter. 

        Pass in the necessary time series depending on which parameter is passed in. E.g. if using beta,
        pass in sigma2 series. 

        Parameters
        ----------
        param : Literal[]
            One of ['omega', 'alpha', 'beta', 'gamma']

        init_value:
            If not passed in, will use unconditional expectation of the parameter.

        x: array of shape (t,), representing one exogenous feature.

        Returns
        -------
        array_like
            array of first partial derivative of sigma2 to param.
        """
        if init_value is None:
            if param == 'omega':
                init_value = 1 / (1 - self.beta)
            elif param == 'alpha' or param == 'beta':
                if x is None:
                    init_value = ((self.omega) 
                              /((1 - self.beta) * (1 - self.alpha - self.beta)))
                else:
                    init_value = ((self.omega + np.sum(self.gamma*np.mean(x**2))) 
                              /((1 - self.beta) * (1 - self.alpha - self.beta)))
            elif param == 'gamma':
                init_value = np.mean(self.x**2) / (1 - self.beta)
            else:
                print('Wrong param value passed into compute_sigma2_first_derivative')
                return None

        res = np.zeros(self.n_obs)
        res[0] = init_value

        for t in range(1, self.n_obs):
            
            if param == 'omega':
                res[t] = 1 + self.beta * res[t-1]
            elif param == 'alpha':
                res[t] = e[t-1]**2 + self.beta * res[t-1]
            elif param == 'beta':
                res[t] = sigma2[t-1] + self.beta * res[t-1]
            elif param == 'gamma':
                res[t] = x[t]**2 + self.beta * res[t-1]
        
        return res

    
    def compute_sigma2_second_derivative(self, param_1: str, param_2: str, init_value=None, first_pd: list=None) -> list:
        """Compute second partial derivative of sigma squared to any two parameters.
        
        Note: Param

        Parameters
        ----------
        param_1 : str
             One of ['omega', 'alpha', 'beta', 'gamma']
        param_2 : str
             One of ['omega', 'alpha', 'beta', 'gamma']
        init_value : _type_, optional
            Second partial derivative value at t=1. If not passed in, will use unconditional expectation of the parameter.
        first_pd:
            Only relevant if one of the params is beta. Pass in the first derivative of the other (non beta) parameter.

        Returns
        -------
        array_like
            array of second partial derivative of sigma2 to both params.
        """
        # If neither param is beta, since the unconditional expectation of the second pd is zero, the whole series is 0.
        if init_value is None and param_1 != 'beta' and param_2 != 'beta':
            res = np.zeros(self.n_obs)
            return res
        
        elif init_value is None:
            if param_2 == 'beta':  # Make param_1 be beta.
                param_1, param_2 = param_2, param_1
            
            if param_2 == 'omega':
                init_value = 1 / (1 - self.beta)**2
            elif param_2 == 'alpha' or param_2 == 'beta':
                if 'gamma' not in self.model_params.keys():
                    init_value = ((self.omega) 
                              /((1 - self.beta)**2 * (1 - self.alpha - self.beta)))
                else:
                    init_value = ((self.omega + np.sum(self.gamma*np.mean(self.x**2)))
                              /((1 - self.beta)**2 * (1 - self.alpha - self.beta)))
            elif param_2 == 'gamma':
                init_value = np.mean(self.x**2) / (1 - self.beta)**2
            else:
                print('Wrong param value passed into compute_sigma2_second_derivative')
                return
             
        res = np.zeros(self.n_obs)
        res[0] = init_value

        for t in range(1, self.n_obs): 
            res[t] = first_pd[t-1] + self.beta * res[t-1]

        return res

    
    def compute_ll_second_derivative(self, e, sigma2, sec_dev, first_dev_p1, first_dev_p2) -> float:
        """Calculate second derivative of the log likelihood to any two parameters, p1 and p2.

        The formula of the second derivative of the log likelihood can be found in our file
        Theoretical Stuff.ipynb. The values of the first partial derivatives to both parameters,
        and the second partial derivative to both parameters, are arrays (representing the derivative 
        of sigma^2 at each time step t) and passed in to this function.

        Parameters
        ----------
        e : array_like 
            e array.
        sigma2 : _type_
            sigma2 array.
        sec_dev : array_like
            second derivative series of sigma^2 to both params.
        first_dev_p1 : array_like
            first derivative series of sigma^2 to the first params.
        first_dev_p2 : array_like
            first derivative series of sigma^2 to the second params.

        Returns
        -------
        float
            calculated value of the second derivative
        """

        # Should flatten the factors before doing assertion
        assert len(sec_dev) == len(first_dev_p1), f"{len(sec_dev)}, {len(first_dev_p1)}"  
        assert len(sec_dev) == len(first_dev_p2), f"{len(sec_dev)}, {len(first_dev_p2)}"  
 

        first_part = sec_dev * (1/sigma2 - e**2 / sigma2**2)
        second_part = first_dev_p1 * first_dev_p2 * (2*e**2/sigma2**3 - 1/sigma2**2)

        res = -0.5*np.sum(first_part + second_part)

        return res
    

    def calculate_information_matrix(self, e, sigma2, x=None, init_value=None):
        """Returns the observed information matrix associated with the log likelihood.

        Parameters
        ----------
        e : _type_
            _description_
        sigma2 : _type_
            _description_
        x : _type_
            _description_

        Returns
        -------
        NDArray
            The information matrix. 
        """
        exo_var_count = 0
        if 'gamma' in self.model_params:
            exo_var_count = self.model_params['gamma'].flatten().shape[0]
        param_count = 1 + self.p + self.q + self.z * exo_var_count
        information_matrix = np.ones((param_count, param_count))

        # Go in the order of omega, alpha, beta, gamma
        params = ['omega']
        params.extend(['alpha']*self.p)
        params.extend(['beta']*self.q)
        params.extend(['gamma']*(self.z*exo_var_count))
        
        first_pd = []  # first pd sigma to each parameter.
        second_pd = [[] for _ in range(param_count)]

        # 1. Compute all first derivative sigma to 4 params. 
        cur_z = 0
        for i in range(param_count):  # can do list comprehension.
            if params[i] == 'gamma':
                first_derivatives = self.compute_sigma2_first_derivative(params[i], e=e, sigma2=sigma2, x=x[:,cur_z], init_value=init_value)
                cur_z += 1
            else:
                first_derivatives = self.compute_sigma2_first_derivative(params[i], e=e, sigma2=sigma2, init_value=init_value)
            first_pd.append(first_derivatives)

        # 2. Compute all second derivative sigma to each param. For 4 params, now we do 16 operations, but can cut to 10.
        for i in range(param_count):
            for j in range(param_count):
                if params[i] == 'beta':
                    second_derivatives = self.compute_sigma2_second_derivative(params[i], params[j], first_pd=first_pd[j],
                                                                               init_value=init_value)
                    second_pd[i].append(second_derivatives)
                elif params[j] == 'beta':
                    second_derivatives = self.compute_sigma2_second_derivative(params[i], params[j], first_pd=first_pd[i],
                                                                               init_value=init_value)
                    second_pd[i].append(second_derivatives)
                else:
                    second_derivatives = self.compute_sigma2_second_derivative(params[i], params[j], init_value=init_value)
                    second_pd[i].append(second_derivatives)

        # 3. Compute information matrix
        for row in range(param_count):
            for col in range(param_count):
                information_matrix[row][col] = -1 * self.compute_ll_second_derivative(e, sigma2, second_pd[row][col], first_pd[row], first_pd[col])

        return information_matrix
    

    def summary(self):
        info_mat_inv = np.linalg.inv(self.information_matrix).diagonal()
        
        index = []
        coef = []
        for k, v in self.model_params.items():
            if type(v) == np.float64:
                index.append(k)
                coef.append(v)
            else:
                v = v.flatten()
                for param_ind, param_val in enumerate(v):
                    index.append(f"{k}[{param_ind}]")
                    coef.append(param_val)

        diagnosis_df = pd.DataFrame(data={'coef': coef, 'std err': np.sqrt(info_mat_inv)}, index=index)
        diagnosis_df['t'] = diagnosis_df['coef'] / diagnosis_df['std err']
        
        df = self.n_obs - (self.p + self.q + self.z + 1)
        diagnosis_df['P>|t|'] = t.sf(abs(diagnosis_df['t']), df=df)

        return diagnosis_df


    def _parse_params(self, params_repam: list, x, as_dict=False):
        """Helper function to parse reparameterized parameters into usable form."""
        omega = self.repam(params_repam[0])
        alpha = self.repam(params_repam[1: self.p + 1])
        beta = self.repam(params_repam[self.p + 1: self.p + self.q + 1])
        gamma = None if x is None else self.repam(params_repam[self.p + self.q + 1:]).reshape(self.z, x.shape[1])

        if as_dict:
            if len(alpha) == 1:
                alpha = alpha[0]
            if len(beta) == 1:
                beta = beta[0]
            params_dict = {'omega': omega, 'alpha': alpha, 'beta': beta, 'gamma': gamma
            } if x is not None else {'omega': omega, 'alpha': alpha, 'beta': beta}

            return params_dict
        
        return omega, alpha, beta, gamma


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


    def compute_sigma2(self,  e: np.array, init_sigma: float=None, x=None):
        sigma2 = np.zeros(self.n_obs)
        if init_sigma is None:
            sigma2[0] = np.var(self.y)
        else:
            sigma2[0] = init_sigma

        for t in range(max(self.p, self.q, self.z), self.n_obs):
            if x is not None:
                sigma2[t] = (self.omega + np.sum(self.alpha * (e[t - self.p : t] ** 2)) 
                                + np.sum(self.beta * (sigma2[t - self.q : t])) 
                                + np.sum(self.gamma * x[t - self.z + 1 : t + 1, :] ** 2)) 
            else:
                sigma2[t] = self.omega + np.sum(self.alpha * (e[t - self.p: t] ** 2)) + np.sum(self.beta * (sigma2[t - self.q: t]))
        return sigma2


    def repam(self, params):
        """Reparameterize parameters for optimization stability."""
        return np.exp(params)

    def inv_repam(self, params):
        """Inverse reparameterization for optimization stability."""
        return np.log(params)



if __name__ == "__main__":
    def generate_data(omega: float, alpha: float, beta: float, gamma: float, T: int = 1000, exo_var_count=1):
        e = np.zeros(T)
        sigma2 = np.zeros(T)
        x = np.random.randn(T, exo_var_count)

        sigma2[0] = 1  

        for t in range(1, T):  
            sigma2[t] = omega + alpha * e[t-1]**2 + beta * sigma2[t-1] + gamma * x[t-1]**2 
            e[t] = np.random.normal(0, np.sqrt(sigma2[t]))

        return e, sigma2, x


    omega, alpha, beta, gamma = 0.1, 0.3, 0.4, 0.2
    e, sigma2, x = generate_data(omega, alpha, beta, gamma, T = 750, exo_var_count=1)

    # Fit using our library. 
    garch = GARCH(p=1, q=1, z=1, verbose=True)
    garch.train(e, x=x)

    print(garch.summary())
    print(garch.loglikelihood)

    # Fit using ARCH library
    from arch import arch_model
    model = arch_model(e, vol='GARCH', mean='zero', p=1, q=1)
    garch_fit = model.fit(disp='off', cov_type='classic')

    print(garch_fit.summary())

