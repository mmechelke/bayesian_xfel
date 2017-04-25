import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod

class AbstractPosterior(object):
    """
    Posterior of a particular volume


    """
    __metaclass__ = ABCMeta

    def __init__(self, likelihood,
                 projection, quadrature,
                 data, prior=None):
        self._likelihood = likelihood
        self._projection = projection
        self._q = quadrature
        self._data = data
        self._prior = prior
        self._params = {}

        if data is not None:
            for i in range(len(data)):
                self._params[i] = self._likelihood.get_params()

    @property
    def params(self):
        return self._params

    def __call__(self, x):
        return self.energy(x)

    @abstractmethod
    def energy(self, x, data=None):
        raise NotImplementedError("Subclass responsability")

    @abstractmethod
    def gradient(self, x, data=None):
        raise NotImplementedError("Subclass responsability")

    @abstractmethod
    def sample_nuissance_parameters(self, x, data):
        raise NotImplementedError("Subclass responsability")


class IntegratedOrientationPosterior(AbstractPosterior):
    """
    Posterior which solves the orientation estimation by
    Integration over all nuisance variables
    """

    def energy(self, x, data=None):
        from scipy.misc import logsumexp

        if data is None:
            data = self._data

        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energies = np.zeros(m)
        energy = 0.

        for d in data:
            for i, s in enumerate(slices):
                energy_ds = self._likelihood.energy(s, d)
                energies[i] = energy_ds
            energy += -logsumexp(-energies)
        if self._prior is not None:
            energy += self._prior.energy(x)

        return energy

    def gradient(self, x, data=None):
        from scipy.misc import logsumexp

        if data is None:
            data = self._data

        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        grad = np.zeros(slices.shape)
        temp_grad = np.zeros(grad.shape)
        energies = np.zeros(m)
        energy = 0.
        for d in data:
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

    def sample_nuissance_parameters(self, x, data):
        raise NotImplementedError("Lazy developer")


class FullPosterior(AbstractPosterior):

    def __init__(self, likelihood,
                 projection, quadrature,
                 data, prior=None):
        super(FullPosterior, self).__init__(likelihood,
                                            projection, quadrature,
                                            data, prior=None)
        self._orientations = np.random.choice(len(self._q.R),
                                              size=len(data))

    def energy(self, x, data=None):
        if data is None:
            data = self._data

        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energy = 0.

        for i, d in enumerate(data):
            self._likelihood.set_params(self._params[i])
            j = self._orientations[i]
            s = slices[j]
            energy += self._likelihood.energy(s, d.ravel())

        if self._prior is not None:
            energy += self._prior.energy(x)

        return energy

    def gradient(self, x, data=None):
        if data is None:
            data = self._data

        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        grad = np.zeros(slices.shape, dtype=x.dtype)

        for i, d in enumerate(data):
            self._likelihood.set_params(self._params[i])
            j = self._orientations[i]
            s = slices[j]
            energy_ds, grad_ds = self._likelihood.gradient(s, d.ravel())
            grad[j,:] += grad_ds

        # project back
        backproj_grad = self._projection.T.dot(grad.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad)).ravel()

        if self._prior is not None:
            backproj_grad += self._prior.gradient(x)
        return backproj_grad


    def sample_nuissance_parameters(self, x, data=None):
        if data is None:
            data = self._data

        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))

        for i, d in enumerate(data):
            self._likelihood.set_params(self._params[i])
            j = self._orientations[i]
            self._likelihood.sample_nuissance_params(slices[j], d.ravel(), self._params[i])

    def sample_rotations(self, x, data=None):
        if data is None:
            data = self._data

        # project density to slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energies = np.zeros((self._data.shape[0],m))

        for i,d in enumerate(data):
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


class MSFullPosterior(AbstractPosterior):

    def __init__(self, likelihood,
                 projection, quadrature,
                 data, prior=None):
        super(MSFullPosterior, self).__init__(likelihood,
                                            projection, quadrature,
                                            data, prior=None)
        self._orientations = np.random.choice(len(self._q.R),
                                              size=len(data))

        self._assignments = np.random.choice(2, size=len(data))


    def energy(self, x, dataset=None):
        if dataset is None:
            dataset = self._data

        xx = x.reshape((2,x.size/2))
        # project density to slices
        m = len(self._q.R)

        slices = [self._projection.dot(xx[i].reshape((-1,))).reshape((m, -1)) for i in range(2)]
        slices = np.squeeze(np.asarray(slices))
        energy = 0.

        for i, d in enumerate(dataset):
            self._likelihood.set_params(self._params[i])
            j = self._orientations[i]
            k = self._assignments[i]
            s = slices[k,j]
            energy += self._likelihood.energy(s, d.ravel())


        return energy

    def gradient(self, x, dataset=None):
        if dataset is None:
            dataset = self._data

        xx = x.reshape((2,x.size/2))
        # project density to slices
        m = len(self._q.R)

        slices = [self._projection.dot(xx[i].reshape((-1,))).reshape((m, -1)) for i in range(2)]
        slices = np.squeeze(np.asarray(slices))
        energy = 0.
        grad = np.zeros(slices.shape, dtype=x.dtype)

        for i, d in enumerate(dataset):
            self._likelihood.set_params(self._params[i])
            j = self._orientations[i]
            k = self._assignments[i]
            s = slices[k,j]
            energy_ds, grad_ds = self._likelihood.gradient(s, d.ravel())
            grad[k,j,:] += grad_ds

        # project back
        backproj_grad = [self._projection.T.dot(grad[k].ravel())
                         for k in range(2)]

        backproj_grad = np.squeeze(np.asarray(backproj_grad)).ravel()

        if self._prior is not None:
            backproj_grad += self._prior.gradient(x)
        return backproj_grad


    def sample_nuissance_parameters(self, x, dataset=None):
        # project density to slices
        if dataset is None:
            dataset = self._data

        xx = x.reshape((2,x.size/2))
        # project density to slices
        m = len(self._q.R)

        slices = [self._projection.dot(xx[i].reshape((-1,))).reshape((m, -1)) for i in range(2)]
        slices = np.squeeze(np.asarray(slices))
        energy = 0.

        for i, d in enumerate(dataset):
            self._likelihood.set_params(self._params[i])
            j = self._orientations[i]
            k = self._assignments[i]
            s = slices[k,j]
            self._likelihood.sample_nuissance_params(s, d.ravel(), self._params[i])


    def sample_rotations(self, x, dataset=None):
        # project density to slices
        if dataset is None:
            dataset = self._data

        xx = x.reshape((2,x.size/2))
        # project density to slices
        m = len(self._q.R)

        slices = [self._projection.dot(xx[i].reshape((-1,))).reshape((m, -1)) for i in range(2)]
        slices = np.squeeze(np.asarray(slices))
        energies = np.zeros((self._data.shape[0],2,m))

        for i,d in enumerate(dataset):
            for k in range(2):
                for j,s in enumerate(slices[k]):
                    if i in self._params:
                        self._likelihood.set_params(self._params[i])
                    energy_ds = self._likelihood.energy(s, d)
                    energies[i,k,j] = energy_ds

        energies = energies.reshape(len(dataset),-1)

        p = np.exp(-energies
                   - sp.misc.logsumexp(-energies,-1)[:,np.newaxis])

        for i in range(len(dataset)):
            # Just to be sure
            p_i = p[i]/p[i].sum()
            best_combi  = np.random.choice(2 * m, p=p_i)
            k,j = np.unravel_index(best_combi, (2,m))
            self._orientations[i] = j
            self._assignments[i] = k
