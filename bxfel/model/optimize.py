import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod

class DataSet(object):

    def __init__(self, data = None, params=None):

        if data is None:
            self._data = []
            self._params = []
        elif params is None:
            self._data = data
            self._params = [{"orientation":np.eye(3)} for _ in range(len(data))]
        else:
            if len(data) != len(params):
                raise ValueError("data and params differ in size")
            self._data = data
            self._params = params

        if len(self._data) != len(self._params):
            raise ValueError("data and params differ in size")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return DataSet(self._data[key], self._params[key])
        elif isinstance(key, tuple):
            return DataSet([self._data[i] for i in key],
                           [self._params[i] for i in key])
        elif isinstance(key, int):
            return self._data[key], self._params[key]
        else:
            raise TypeError("Illegal index")

    @property
    def data(self):
        return self._data

    @property
    def parameter(self):
        return self._params

    def __iter__(self):
        for i in range(len(self._data)):
            yield self._data[i], self._params[i]


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



class Prior(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def energy(self, x):
        raise NotImplementedError("Not implemented in Base class")

    @abstractmethod
    def gradient(self, x):
        raise NotImplementedError("Not implemented in Base class")


class ExponentialPrior(Prior):

    def __init__(self, scale=1):
        self._lambda = scale

    def energy(self, x):
        pass

    def gradient(self, x):
        pass

class LaplacePrior(Prior):

    def __init__(self, k=1.):
        self._k = k

    def energy(self, x):
        return self._k * np.sum(np.abs(x))

    def gradient(self, x):
        return self._k * np.sign(x)

class DoubleExpPrior(Prior):

    """
    Double exponential prior with two parameters
    controlling the rate for positive and negative
    values independently
    """

    def __init__(self, lambda_neg, lambda_pos):
        self._lambda_neg = lambda_neg
        self._lambda_pos = lambda_pos

    def energy(self, x):
        ln = self._lambda_neg
        lp = self._lambda_pos
        x_pos = x[x>0]
        x_neg = x[x<0]

        u_pos = -len(x_pos) * np.log(lp)  +  lp * np.sum(x_pos)
        u_neg = -len(x_neg) * np.log(ln)  +  ln * np.sum(-x_neg)

        return u_pos + u_neg

    def gradient(self, x):
        grad = np.zeros_like(x)
        grad[x>0] = self._lambda_pos
        grad[x<0] = -self._lambda_neg

        return grad




class LocalMeanPrior(Prior):

    def __init__(self, k, N):
        """
        assumes that x is in fact
        a N x N x N Tensor
        """
        self._k = k
        self._N = int(N)


    def energy(self, x):
        tmp = x.reshape((self._N, self._N, self._N))
        u = 0.0
        for i in range(self._N):
            for j in range(self._N):
                for k in range(self._N):

                    for l in [-1, 0, 1]:
                        for m in [-1, 0, 1]:
                            for n in [-1, 0, 1]:
                                if l==0 and m==0 and n==0:
                                    continue
                                if (i+l >= 0 and i+l< self._N
                                    and j+m >= 0 and j+m < self._N
                                    and k+n >= 0 and k+n < self._N):
                                    u += 0.5 * self._k * (tmp[i,j,k] - tmp[i+l,j+m,k+n])**2
        return u

    def gradient(self, x):
        tmp = x.reshape((self._N, self._N, self._N))
        grad = np.zeros_like(tmp)

        u = 0.0
        for i in range(self._N):
            for j in range(self._N):
                for k in range(self._N):

                    for l in [-1, 0, 1]:
                        for m in [-1, 0, 1]:
                            for n in [-1, 0, 1]:
                                if l==0 and m==0 and n==0:
                                    continue
                                if (i+l >= 0 and i+l< self._N
                                    and j+m >= 0 and j+m < self._N
                                    and k+n >= 0 and k+n < self._N):
                                    grad[i,j,k] +=  self._k * (tmp[i,j,k] - tmp[i+l,j+m,k+n])
                                    grad[i+l,j+m,k+n] -=  self._k * (tmp[i,j,k] - tmp[i+l,j+m,k+n])
        return grad.ravel()


class SimpleOpt(object):

    def __init__(self, likelihood, projection, quadrature, premult, data):
        self._likelihood = likelihood
        self._projection = projection
        self._q = quadrature
        self._premultipier = premult
        self._data = data

    def energy(self, x):
        # project density to slices
        slices = self._projection.dot(x.reshape((-1,)))
        return self._likelihood.energy(slices, self._data)

    def gradient(self, x):
        # project density to slices
        slices = self._projection.dot(x.reshape((-1,)))
        energy, grad = self._likelihood.gradient(slices, self._data)

        # project back
        backproj_grad = self._projection.T.dot(grad)
        return energy, backproj_grad/(self._premultipier + 1)

    def gradient2(self, x):
        # project density to slices
        slices = self._projection.dot(x.reshape((-1,)))
        energy, grad = self._likelihood.gradient(slices, self._data)

        # project back
        backproj_grad = self._projection.T.dot(grad)
        return energy,  backproj_grad


class Objective(object):

    def __init__(self, likelihood, projection, quadrature,
                 premult, data, prior=None):
        self._likelihood = likelihood
        self._projection = projection
        self._q = quadrature
        self._premultipier = premult
        self._data = data
        self._prior = prior

    def gradient(self, x):
        from scipy.misc import logsumexp
        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        grad = np.zeros(slices.shape)
        temp_grad = np.zeros(grad.shape)
        energies = np.zeros(m)
        energy = 0.
        for d in self._data:
            for i,s in enumerate(slices):
                energy_ds, grad_ds = self._likelihood.gradient(s, d)
                temp_grad[i,:] = grad_ds
                energies[i] = energy_ds

            
            energy += energies.sum()

            temp_grad = temp_grad * np.exp(-energies-logsumexp(-energies))[:,np.newaxis]
            grad += temp_grad
            
        # project back
        backproj_grad = self._projection.T.dot(grad.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad))

        if self._prior is not None:
            backproj_grad += self._prior.gradient(x)
        return backproj_grad

    def __call__(self, x):
        return self.energy(x)

    def energy(self, x, data=None):
        from scipy.misc import logsumexp
        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energies = np.zeros(m)
        energy = 0.

        for d in self._data:
            for i, s in enumerate(slices):
                energy_ds = self._likelihood.energy(s, d)
                energies[i] = energy_ds
            energy += -logsumexp(-energies)
        if self._prior is not None:
            energy += self._prior.energy(x)

        return energy



class GibbsOrientationSampler(object):

    def __init__(self, likelihood, projection, quadrature,
                 data, prior=None, params=None):
        """
        @param alpha: prior hyperparameter of the prior

        """
        self._likelihood = likelihood
        self._projection = projection
        self._q = quadrature
        self._data = data
        self._prior = prior
        self._orientations = np.random.choice(len(self._q.R),
                                              size=len(data))

        if params is None:
            self._params = {}
        else:
            self._params = params

    def energy(self, x):
        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energy = 0.

        for i, d in enumerate(self._data):
            if i in self._params:
                self._likelihood.set_params(self._params[i])

            j = self._orientations[i]
            s = slices[j]
            energy += self._likelihood.energy(s, d.ravel())

        if self._prior is not None:
            energy += self._prior.energy(x)

        return energy

    def gradient(self, x):
        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        grad = np.zeros(slices.shape, dtype=x.dtype)

        for i, d in enumerate(self._data):
            if i in self._params:
                self._likelihood.set_params(self._params[i])
            j = self._orientations[i]
            s = slices[j]
            energy_ds, grad_ds = self._likelihood.gradient(s, d.ravel())
            grad[j,:] += grad_ds

        # project back
        backproj_grad = self._projection.T.dot(grad.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad))

        if self._prior is not None:
            backproj_grad += self._prior.gradient(x)
        return backproj_grad

    def __call__(self, x):
        return self.energy(x)

    def update_params(self, x):
        for i,d in enumerate(self._data):
            params = self._params[i]
            self._likelihood.update_params(x,d, params)

    def update_rotations(self, x, sample_params=False):
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energies = np.zeros((self._data.shape[0],m))

        for i,d in enumerate(self._data):
            for j,s in enumerate(slices):
                if i in self._params:
                    self._likelihood.set_params(self._params[i])
                energy_ds = self._likelihood.energy(s, d)
                energies[i,j] = energy_ds

        p = np.exp(-energies
                   - sp.misc.logsumexp(-energies,-1)[:,np.newaxis])
        for i in range(len(self._data)):
            # Just to be sure
            p_i = p[i]/p[i].sum()
            self._orientations[i] = np.random.choice(m, p=p_i)

        if sample_params:
            for i, d in enumerate(self._data):
                old_params = self._params[i]
                j = self._orientations[i]
                s = slices[j]
                self._likelihood.sample_nuissance_params(s, d, old_params)

    def run(self, x0, niter, eps=1e-2, sample_params=False, skip=-1, verbose=20):
        from csbplus.mcmc import HMC
        self.update_rotations(x0)
        self._samples = []
        self._energies = []
        hmc = HMC(potential_energy=self,
                  nsteps=20, eps=eps)
        hmc._adaptphase = 1.

        for i in range(niter):
            s = hmc.run(x0, 5, return_momenta=False)
            x0 = s[-1]
            self.update_rotations(x0, sample_params)

            if skip == -1:
                self._samples.extend(s)
                self._energies.extend(map(self.energy, s))
            elif i%int(skip)==0 or i == niter-1:
                self._samples.extend(s)
                self._energies.extend(map(self.energy, s))

            if i% verbose == 0:
                print "iteration: {}, energy: {}, stepsize: {}".format(i, self._energies[-1], hmc._eps[-1])

        return self._energies, self._samples


class GibbsBackProjection(GibbsOrientationSampler):

    def __init__(self, likelihood, projection, quadrature,
                 data, premult, prior=None, params=None):
        super(GibbsBackProjection, self).__init__(likelihood, projection,
                                                  quadrature, data,
                                                  prior, params)
        self._premult = premult
        # concentration prior
        self._alpha = 0.0

    def _back_project(self):
        m = len(self._q.R)
        slices = np.zeros(self._projection.shape[0])
        slices = slices.reshape((m, -1))

        counts  = np.zeros(m)

        for i, d in enumerate(self._data):
            j = self._orientations[i]
            counts[j] += 1
            slices[j,:] += d.ravel()

        # project back
        new_volume = self._projection.T.dot(slices.ravel())

        # Now normalize this our volume

        count_slices = np.outer(counts, np.ones(self._data.shape[1])).ravel()
        denominator = self._projection.T.dot(count_slices)
        new_volume[denominator.nonzero()] /= denominator[denominator.nonzero()]

        return new_volume

    def update_rotations(self, x, sample_params=False):
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energies = np.zeros((self._data.shape[0],m))

        for i,d in enumerate(self._data):
            for j,s in enumerate(slices):
                if i in self._params:
                    self._likelihood.set_params(self._params[i])
                energy_ds = self._likelihood.energy(s, d)
                energies[i,j] = energy_ds

        p = np.exp(-energies
                   - sp.misc.logsumexp(-energies,-1)[:,np.newaxis])
        for i in range(len(self._data)):
            # Just to be sure
            p_i = (p[i] + self._alpha) / (p[i].sum() + self._alpha * p.shape[1])
            self._orientations[i] = np.random.choice(m, p=p_i)

        if sample_params:
            for i, d in enumerate(self._data):
                old_params = self._params[i]
                j = self._orientations[i]
                s = slices[j]
                self._likelihood.sample_nuissance_params(s, d, old_params)


    def run(self, x0, niter, sample_params=False, skip=-1, verbose=20):
        from csbplus.mcmc import HMC
        self.update_rotations(x0)

        self._samples = []
        self._energies = []

        for i in range(niter):
            x = self._back_project()
            self.update_rotations(x)

            if skip == -1:
                self._samples.append(x)
                self._energies.append(self.energy(x))
            elif i%int(skip)==0 or i == niter-1:
                self._samples.append(x)
                self._energies.append(self.energy(x))

            if i% verbose == 0:
                print "iteration: {}, energy: {}".format(i, self._energies[-1])

        return self._energies, self._samples


class GibbsSGD(object):

    def __init__(self, likelihood, projection, quadrature,
                 data, prior=None, params=None,
                 batchsize=500):
        """
        @param alpha: prior hyperparameter of the prior

        """
        self._likelihood = likelihood
        self._projection = projection
        self._q = quadrature
        self._prior = prior
        self._orientations = np.random.choice(len(self._q.R),
                                              size=len(data))
        if params is None:
            params = [{"orientation":self._orientations[i]}
                      for i in range(len(data))]
        self._data = DataSet(data, params)

    def energy(self, x, data=None):
        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energy = 0.
        if data is None:
            data = self._data
        for i, (d, p) in enumerate(data):
            j = p["orientation"]
            s = slices[j]
            energy += self._likelihood.energy(s, d.ravel())

        return energy

    def gradient(self, x):
        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        grad = np.zeros(slices.shape)

        for i, (d,p) in enumerate(self._data):
            self._likelihood.set_params(p)
            j = p["orientation"]
            s = slices[j]
            energy_ds, grad_ds = self._likelihood.gradient(s, d.ravel())
            grad[j,:] += grad_ds

        # project back
        backproj_grad = self._projection.T.dot(grad.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad))

        return backproj_grad

    def __call__(self, x):
        return self.energy(x)

    def update_params(self, x):
        for i, (d,p) in enumerate(self._data):
            self._likelihood.update_params(x, d, p)

    def update_rotations(self, x, sample_params=False):
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energies = np.zeros((self._data.shape[0],m))

        for i,d in enumerate(self._data):
            for j,s in enumerate(slices):
                if i in self._params:
                    self._likelihood.set_params(self._params[i])
                energy_ds = self._likelihood.energy(s, d)
                energies[i,j] = energy_ds

        p = np.exp(-energies
                   - sp.misc.logsumexp(-energies,-1)[:,np.newaxis])
        for i in range(len(self._data)):
            # Just to be sure
            p_i = p[i]/p[i].sum()
            self._orientations[i] = np.random.choice(m, p=p_i)

        if sample_params:
            for i, d in enumerate(self._data):
                old_params = self._params[i]
                j = self._orientations[i]
                s = slices[j]
                self._likelihood.sample_nuissance_params(s, d, old_params)

if __name__ == "__main__":
    n = 10
    eps = 1e-7
    x = np.random.gamma(10, 10, n**3)
    p = LocalMeanPrior(1., n)
    u = p.energy(x)
    g = p.gradient(x)

    num_grad = np.zeros(g.shape)

    eps = 1e-6

    for i in range(x.size):
        x[i] += eps
        num_grad[i] = (p.energy(x) - u)/eps
        x[i] -= eps

