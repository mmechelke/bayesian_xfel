import numpy as np
import scipy as sp

import abc

import ctypes
import multiprocessing as mp


from xfel.grid.optimize import GaussianLikelihood
# Global variables for multiprocessing
shared_slices = None
shared_data = None
shared_gradient = None
shared_gradient_base  = None


def do_energy(params):
    slice_index, data_index = params
    global shared_slices
    global shared_data

    ll = GaussianLikelihood(1.)
    s = shared_slices[slice_index]
    d = shared_data[data_index]

    return ll.energy(s,d)

def do_gradient(params):
    slice_index, data_index = params
    global shared_slices
    global shared_data
    global shared_gradient
    global shared_gradient_base

    ll = GaussianLikelihood(1.)
    s = shared_slices[slice_index]
    d = shared_data[data_index]
    energy, grad = ll.gradient(s,d)

    with shared_gradient_base.get_lock():
        shared_gradient[slice_index] += grad
 
    return energy

def do_energy_and_gradient(params):
    slice_index, data_index = params
    global shared_slices
    global shared_data

    ll = GaussianLikelihood(1.)
    s = shared_slices[slice_index]
    d = shared_data[data_index]

    return ll.gradient(s,d)



class Model(object):

    def __init__(self, likelihood, projection, quadrature, data, prior=None,
                 update_likelihoods=False):
        self._likelihood = likelihood
        self._projection = projection
        self._q = quadrature
        self._data = data
        self._prior = prior

        self._m = len(quadrature.R)
        self._n = len(data)

    def __call__(self, x):
        return self.energy(x)

    def energy(self, x):
        return self.batch_energy(x, range(self._n))

    @abc.abstractmethod
    def batch_energy(self , x, indices):
        pass

    def gradient(self, x):
        return self.batch_gradient(x, range(self._n))

    @abc.abstractmethod
    def batch_gradient(self , x, indices):
        pass

    def _project_slices(self, x):
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((self._m, -1))
        slices = np.squeeze(np.asarray(slices))
        return slices


class IntegrateOrientation(Model):

    def batch_energy(self, x, indices):
        from scipy.misc import logsumexp
        slices = self._project_slices(x)
        energy = 0.
        for j in indices:
            energies = np.zeros(self._m)
            for i, s in enumerate(slices):
                energy_ds = self._likelihood.energy(s, self._data[j])
                energies[i] = energy_ds
            energy +=  -logsumexp(-energies)
        if self._prior is not None:
            energy += self._prior.energy(x)
        return energy

    def batch_gradient(self, x, indices):
        from scipy.misc import logsumexp
        slices = self._project_slices(x)
        energy = 0.
        grad = np.zeros(slices.shape)
        temp_grad = np.zeros(grad.shape)
        energies = np.zeros(self._m)
        energy = 0.
        for j in indices:
            for i,s in enumerate(slices):
                energy_ds, grad_ds = self._likelihood.gradient(s, self._data[j])
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


    def update_params(self, x):
        pass


class SampleOrientation(Model):

    def __init__(self, likelihood, projection,
                 quadrature, data, prior=None,
                 update_likelihoods=False):
        super(SampleOrientation, self).__init__(likelihood, projection,
                                                quadrature, data, prior)
        self._orientations = np.random.choice(len(self._q.R),
                                              size=len(data))

        self._update_rotations = True
        self._update_likelihoods = update_likelihoods
        self._params = {}
        if update_likelihoods:
            for i in range(len(self._data)):
                self._params[i] = self._likelihood.get_params()

    def batch_energy(self, x, indices):
        from scipy.misc import logsumexp
        slices = self._project_slices(x)
        energy = 0.
        for j in indices:
            self._likelihood.set_params(self._params[j])
            i = self._orientations[j]
            energy += self._likelihood.energy(slices[i], self._data[j])
        if self._prior is not None:
            energy += self._prior.energy(x)
        return energy

    def batch_gradient(self, x, indices):
        from scipy.misc import logsumexp
        slices = self._project_slices(x)
        grad = np.zeros(slices.shape)
        energy = 0.

        for j in indices:
            self._likelihood.set_params(self._params[j])
            i = self._orientations[j]
            energy_di, grad_di = self._likelihood.gradient(slices[i], self._data[j])
            grad[i] += grad_di

        # project back
        backproj_grad = self._projection.T.dot(grad.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad))

        if self._prior is not None:
            backproj_grad += self._prior.gradient(x)
        return backproj_grad

    def update_params(self, x):
        slices = self._project_slices(x)

        if self._update_rotations:
            self.update_rotations(slices)

        if self._update_likelihoods:
            self.update_likelihoods(slices)

    def update_likelihoods(self, slices):

        for i in range(len(self._data)):
            j = self._orientations[i]
            self._likelihood.set_params(self._params[i])
            self._likelihood.sample_nuissance_params(slices[j], self._data[i], self._params[i])

    def update_rotations(self, slices):
        m = len(self._q.R)

        energies = np.zeros((self._data.shape[0], m))

        for i,d in enumerate(self._data):
            for j,s in enumerate(slices):
                self._likelihood.set_params(self._params[i])
                energy_ds = self._likelihood.energy(s, d)
                energies[i,j] = energy_ds

        p = np.exp(-energies
                   - sp.misc.logsumexp(-energies,-1)[:,np.newaxis])
        for i in range(len(self._data)):
            # Just to be sure
            p_i = p[i]/p[i].sum()
            self._orientations[i] = np.random.choice(m, p=p_i)


class MPBase(Model):

    def __init__(self, likelihood, projection,
                 quadrature, data, prior=None,
                 n_cpu=1):

        super(MPBase, self).__init__(likelihood, projection, quadrature, data, prior)
        self._init__process(n_cpu)


    def _init__process(self, n_cpu):
        """
        Create populate global data
        """
        global shared_slices
        global shared_data
        global shared_gradient
        global shared_gradient_base

        shared_slices_base = mp.Array(ctypes.c_double,
                                      self._projection.shape[0],
                                      lock=False)
        shared_slices = np.frombuffer(shared_slices_base,
                                      dtype="double")
        shared_slices = shared_slices.reshape((len(self._q.R),-1))

        shared_gradient_base = mp.Array(ctypes.c_double,
                                        self._projection.shape[0],
                                        lock=True)
        shared_gradient = np.ctypeslib.as_array(shared_gradient_base.get_obj())
        shared_gradient = shared_slices.reshape((len(self._q.R),-1))

        shared_data_base = mp.Array(ctypes.c_double, self._data.size,
                                    lock=False)
        shared_data = np.frombuffer(shared_data_base,
                                    dtype="double")
        shared_data = shared_data.reshape(self._data.shape)
        shared_data[:] = self._data

        self._pool = mp.Pool(n_cpu)


class IntegrateOrientationmp(MPBase):

    def batch_energy(self, x, indices):
        from scipy.misc import logsumexp
        slices = self._project_slices(x)
        global shared_slices
        shared_slices[:] = slices[:]

        params = [(j, i) for j in range(self._m)
                  for i in  indices]
        energies = np.array(self._pool.map(do_energy, params))
        energies = energies.reshape((self._m, self._n))
        return np.sum(-logsumexp(-energies,-1))

    def batch_gradient(self, x, indices):
        raise NotImplementedError("Lazy bastard")


class SampleOrientationmp(MPBase):

    def __init__(self, likelihood, projection,
                 quadrature, data, prior=None, n_cpu=1):
        super(SampleOrientationmp, self).__init__(likelihood, projection,
                                                  quadrature, data, prior, n_cpu=n_cpu)
        self._orientations = np.random.choice(len(self._q.R),
                                              size=len(data))

    def batch_energy(self, x, indices):
        from scipy.misc import logsumexp
        slices = self._project_slices(x)
        global shared_slices
        shared_slices[:] = slices[:]

        params = [(self._orientations[i], i)
                  for i in indices]
        energies = self._pool.map_async(do_energy, params)
        energy = 0.0
        # Maybe we have something to do while we wait
        if self._prior is not None:
            energy = self._prior.energy(x)
        energy += np.sum(energies.get())
        return energy

    def batch_gradient(self, x, indices):
        from scipy.misc import logsumexp
        slices = self._project_slices(x)
        grad = np.zeros(slices.shape)
        energy = 0.

        for j in indices:
            i = self._orientations[j]
            energy_di, grad_di = self._likelihood.gradient(slices[i], self._data[j])
            grad[i] += grad_di

        # project back
        backproj_grad = self._projection.T.dot(grad.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad))

        if self._prior is not None:
            backproj_grad += self._prior.gradient(x)
        return backproj_grad
    
    def __batch_gradient(self, x, indices):
        """
        Currently not working ->
        """
        from scipy.misc import logsumexp
        slices = self._project_slices(x)
        global shared_slices
        global shared_gradient
        shared_slices[:] = slices[:]

        params = [(self._orientations[i], i)
                  for i in indices]
        energies = self._pool.map(do_gradient, params)
        if self._prior is not None:
            prior_grad = self._prior.gradient(x)

        backproj_grad = self._projection.T.dot(shared_gradient.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad))
        if self._prior is not None:
            backproj_grad += prior_grad
        return backproj_grad

    def update_params(self, x):
        m = len(self._q.R)

        slices = self._project_slices(x)
        global shared_slices
        shared_slices[:] = slices[:]
        params = [(j, i)
                  for i in range(self._n)
                  for j in range(self._m)]

        energies = self._pool.map(do_energy, params)
        energies = np.array(energies).reshape((self._n, self._m))

        p = np.exp(-energies
                   - sp.misc.logsumexp(-energies,-1)[:,np.newaxis])

        for i in range(len(self._data)):
            # For the moment let's trust floating point arithmetic
            # p_i = p[i]/p[i].sum()
            self._orientations[i] = np.random.choice(m, p=p[i])
