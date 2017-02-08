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
            if i in self._params:
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


