import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod

class Likelihood(object):
    __metaclass__ = ABCMeta

    def __init__(self, n=1, mask=None):
        self._n = n
        self._mask = None
        if mask is None:
            self._n = n
        else:
            self._n = mask.size - mask.sum()

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value
        if value is not None:
            self._n = value.size - value.sum()

    def _prior_energy(self):
        return 0.0

    def energy(self, theta, data):
        e_prior = self._prior_energy()
        if self._mask is None:
            return self._energy(theta, data) + e_prior
        else:
            return self._energy_mask(theta, data, self._mask) + e_prior

    @abstractmethod
    def _energy(self, theta, data):
        raise NotImplementedError("Not implemented in Base class")

    def _energy_mask(self, theta, data, mask):
        return self._energy(theta[np.logical_not(mask)],
                            data[np.logical_not(mask)])

    def gradient(self, theta, data):
        e_prior = self._prior_energy()
        if self._mask is None:
            energy, gradient =  self._gradient(theta, data)
        else:
            energy, gradient = self._gradient_mask(theta, data, self._mask)

        return energy + e_prior, gradient

    @abstractmethod
    def _gradient(self, theta, data):
        raise NotImplementedError("Not implemented in Base class")

    def _gradient_mask(self, theta, data, mask):
        grad = np.zeros_like(theta)
        indices = np.where(np.logical_not(mask))
        energy, masked_grad = self._gradient(theta[indices],
                                             data[indices])
        grad[indices] = masked_grad
        return energy, grad

    def set_params(self, p):
        if 'mask' in p:
            self._mask = p['mask']
            self._n = self._mask.size - self._mask.sum()

    def get_parms(self, p):
        if mask is not None:
            return self._mask

    def sample_nuissance_params(self, calc_data, data, parameter_dict):
        pass


class GaussianLikelihood(Likelihood):
    """ 
    We assume a scale free prior on gamma
    and a gamma distribution on k
    """

    def __init__(self, k=1., n=1, mask=None):
        super(GaussianLikelihood, self).__init__(n, mask)
        self._k = np.float(k)
        self._gamma = 1.

        self._normal_alpha = 10.
        self._normal_beta = 10.

        self._sample_gamma = False
        self._sample_k = False

    def _prior_energy(self):
        k = self._k
        # Gamma prior on k
        energy = self._normal_beta * k + (self._normal_alpha - 1) * np.log(k)
        # Jeffrey's prior on gamma (improper)
        energy += np.log(self._gamma)
        return energy

    def _energy(self, theta, data):
        n_data = self._n
        chi2 = (data - self._gamma * theta)**2
        E = 0.5 * self._k * chi2.sum()
        # Partion function
        E -= 0.5 * n_data * np.log(self._k)
        return E

    def _gradient(self, theta, data):
        n_data = self._n
        diff = (data - self._gamma * theta)
        energy = 0.5 * self._k * np.sum(diff**2)
        # Partion function
        energy -= 0.5 * n_data * np.log(self._k)
        grad = -self._k * diff
        return energy, grad

    def sample_nuissance_params(self, calc_data, data, parameter_dict):
        if self._sample_k:
            self.sample_force_constant(calc_data, data, parameter_dict)
        if self._sample_gamma:
            self.sample_scale(calc_data, data, parameter_dict)
  
    def sample_scale(self, calc_data, data, parameter_dict):
        """
        Sample a new scale (fluence) gamma
        """
        gamma = self._gamma
        k = self._k

        xy = k * np.sum(data * calc_data)
        xx = k * np.sum(calc_data**2)

        mean = xy / xx
        var  = 1. / xx

        gamma = np.random.normal(mean, np.sqrt(var))
        parameter_dict["gamma"] = gamma


    def sample_force_constant(self,  calc_data, data, parameter_dict):
        """
        Sample a new force constant k
        We assume a gamma prior G(10, 10) on k
        """
        gamma = parameter_dict["gamma"] 
        n = data.size
        chi2   =  np.sum((gamma * calc_data - data)**2)
        alpha = 0.5 * n + self._normal_alpha
        beta = 0.5 * chi2 + self._normal_beta
        parameter_dict['k'] = np.clip(np.random.gamma(alpha, 1./beta), 0.1, 1e5)

    def set_params(self, p):
        super(GaussianLikelihood, self).set_params(p)
        if 'k' in p:
            self._k = np.float(p['k'])
        if "gamma" in p:
            self._gamma = np.float(p['gamma'])

    def get_params(self):
        return {"k": self._k,
                "gamma": self._gamma}

class PoissonLikelihood(Likelihood):
    """
    Poisson Error Model
    Assumes Poisson distributed data
    """

    def _energy(self, theta, data):
        eps = 1e-300
        energy = -data * np.log(np.clip(theta,1e-20, 1e300)) + data * np.log(np.clip(data,1e-20,1e300)) + theta
        return energy.sum()

    def _gradient(self, theta, data):
        energy = -data * np.log(np.clip(theta,1e-20, 1e300)) + data * np.log(np.clip(data,1e-20,1e300)) + theta
        # agressive clipping, maybe I am making the gradient problematic
        grad = 1 - data/np.clip(theta, 1e-10, 1e300)
        return energy.sum(), grad

class LogNormalLikelihood(Likelihood):

    def __init__(self, k=1., n=1, mask=None):
        super(LogNormalLikelihood, self).__init__(n, mask)
        self._k = np.float(k)
        self._gamma = 1.

    def _energy(self, theta, data):
        a = np.log(np.clip(theta,
                           1e-105, 1e300))
        b = np.log(np.clip(data,
                           1e-105, 1e300))

        chi = a - b
        return 0.5 * self._k * np.sum(chi*chi)

    def _gradient(self, theta, data):
        a = np.log(np.clip(theta,
                           1e-105, 1e300))
        b = np.log(np.clip(data,
                           1e-105, 1e300))
        chi = a - b
        return 0.5 * self._k * chi / a



class TiedGaussianLikelihood(Likelihood):
    """
    A likelihood in which the we assume that mean and variance are tied
    plus some constant gaussian noise so the model looks like
    so d_i = x_i + N(0, sqrt(x_i)) +  N(0,sigma)
    """

    def __init__(self, sigma=1., n=1, mask=None):
        super(TiedGaussianLikelihood, self).__init__(n,mask)
        self._sigma = sigma

    def _energy(self, theta, data):
        chi2 = (data - theta)**2
        sqrt_theta = np.sqrt(np.clip(theta, 1e-30, 1e100))

        u  =  np.log(sqrt_theta + self._sigma)\
             + 0.5 * chi2/(sqrt_theta + self._sigma)**2

        return u.sum() - 0.5 * self._n * np.log(2 * np.pi)

    def _gradient(self, theta, data):
        ctheta = np.clip(theta, 1e-30, 1e100)
        diff = (data - theta)
        chi2 = diff**2
        sqrt_theta = np.sqrt(ctheta)
        sqrt_plus_sigma = (sqrt_theta + self._sigma)

        grad =  1./(ctheta + sqrt_theta * self._sigma) * 0.5

        grad -= chi2/(2 * sqrt_theta * sqrt_plus_sigma**3) +  diff/sqrt_plus_sigma**2

        energy =  np.log(sqrt_plus_sigma)\
                 + 0.5 * chi2/(sqrt_plus_sigma)**2
        energy = energy.sum() - 0.5 * self._n * np.log(2 * np.pi)

        return energy, grad





class AnscombeLikelihood(Likelihood):
    """
    Instead of estimating the underlying image we estimate the anscombe transformed
    image

    """
    def __init__(self, k=1., n=1, mask=None):
        super(AnscombeLikelihood, self).__init__(n, mask)
        self._n = n

        self._k = np.float(k)
        self._sigma = 1.

    def _energy(self, theta, data):
        n_data = self._n
        transformed_counts = 2 * np.sqrt(np.clip(data, 0.0, 1e309) + 0.375)
        chi2 = np.sum((transformed_counts.ravel()
                       - theta.ravel())**2)

        return 0.5 * self._k * chi2 - 0.5 * n_data * np.log(self._k)

    def _gradient(self, theta, data):
        n_data = self._n
        transformed_counts = 2 * np.sqrt(np.clip(data, -0.3, 1e309) + 0.375)
        diff = transformed_counts - theta
        chi2 = np.sum((diff)**2)

        energy = 0.5 * self._k * chi2 - 0.5 * n_data * np.log(self._k)

        return energy,  self._k * -diff
