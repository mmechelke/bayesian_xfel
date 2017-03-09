import numpy as np
import scipy as sp

from xfel.grid.optimize import DataSet 
from xfel.utils import chunks


class GibbsSGD(object):

    def __init__(self, likelihood, projection, quadrature,
                 data, prior=None, params=None,
                 eps=1e-3, decay_rate=1e-2,
                 batchsize=500):
        """
        @param alpha: prior hyperparameter of the prior

        """
        self._batchsize = int(batchsize)
        self._initial_learning_rate = float(eps)
        self._decay_rate = decay_rate

        self._likelihood = likelihood
        self._projection = projection
        self._q = quadrature
        self._prior = prior

        self._current_epoch = None
        self._orientations = np.random.choice(len(self._q.R),
                                              size=len(data))
        if params is None:
            params = [{"orientation":self._orientations[i]}
                      for i in range(len(data))]
        self._data = DataSet(data, params)


    def energy(self, x, data=None):
        """
        Evaluate the energy of the 3D volume expressed as vector x

        Note this iterates over all datapoints and will be slow
        """
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energy = 0.
        if data is None:
            data = self._data
        for i, (d, p) in enumerate(data):
            j = self._orientations[i]
            s = slices[j]
            energy += self._likelihood.energy(s, d.ravel())

        return energy

    def gradient(self, x):
        """
        Evaluate the gradient of the 3D volume expressed as vector x

        Note this iterates over all datapoints and will be slow
        """
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

    def _batch_gradient(self, x , batch):
        """
        Evaluate the gradient of the 3D volume expressed as vector x

        """

        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        grad = np.zeros(slices.shape)

        for i in batch:
            d,p = self._data[i]
            self._likelihood.set_params(p)
            j = self._orientations[i]
            s = slices[j]
            energy_ds, grad_ds = self._likelihood.gradient(s, d.ravel())
            grad[j,:] += grad_ds

        # project back
        backproj_grad = self._projection.T.dot(grad.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad))

        return backproj_grad

    def _batch_energy(self, x, batch):
        """
        Evaluate the energy of a batch
        """
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        energy = 0.

        for i in batch:
            d,p = self._data[i]
            j = self._orientations[i]
            s = slices[j]
            energy += self._likelihood.energy(s, d.ravel())

        return energy

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
        energies = np.zeros((len(self._data),m))

        for i, (d, p) in enumerate(self._data):
            for j,s in enumerate(slices):
                try:
                    if i in self._params:
                        self._likelihood.set_params(self._params[i])
                except:
                    pass
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
                j = self._orientations[i]
                s = slices[j]
                self._likelihood.sample_nuissance_params(s, d)



    def run(self, x0, n_epochs, batchsize=None, verbose=20):
        """
        Run stochastic gradient decent of n_epochs epochs

        @returns the samples and energies after each epoch
        """
        if batchsize is None:
            batchsize = self._batchsize

        samples = [x0,]
        energies = []
        t = 0
        x = 1. * x0
        
        # Initial update 
        self.update_rotations(x, sample_params=False)

        n_data = len(self._data)

        for epoch in range(n_epochs):
            batches = chunks(np.random.permutation(n_data), batchsize)

            for batch in batches:

                learning_rate = self._initial_learning_rate / (1 + self._initial_learning_rate * self._decay_rate * t)
                grad =  self._batch_gradient(x, batch)
                x -= learning_rate * grad
                t += 1

            samples.append(x)
            energies.append(self.energy(x))

            self.update_rotations(x, sample_params=False)
            print learning_rate, np.linalg.norm(grad)
        return samples, energies


class GibbsAda(GibbsSGD):

    def run(self, x0, n_epochs, batchsize=None, verbose=20):
        """
        Run stochastic gradient decent of n_epochs epochs

        @returns the samples and energies after each epoch
        """
        if batchsize is None:
            batchsize = self._batchsize

        energies = []

        n_data = len(self._data)
        xs = [x0.copy()]
        x = x0.copy()
        energies = []
        history = np.zeros_like(x) + 1.

        # Initial update 
        self.update_rotations(x, sample_params=False)

        for epoch in xrange(n_epochs):
            batch = np.random.permutation(n_data)

            for i in range(0,n_data-batchsize ,batchsize):
                grad =  self._batch_gradient(x, batch)  
                history += grad **2 
                x -= 1e1 * grad / np.sqrt(history)

            xs.append(x)
            energies.append(self.energy(x))

            self.update_rotations(x, sample_params=False)
            print np.linalg.norm(history), np.linalg.norm(grad)
        return xs, energies


class GibbsNesterov(GibbsSGD):
    
    def __init__(self,  likelihood, projection, quadrature,
                 data, prior=None, params=None,
                 eps=1e-3, decay_rate=1e-2,
                 batchsize=500,
                 gamma = 0.9):
        super(GibbsNesterov, self).__init__(likelihood, projection, quadrature,
                                            data, prior, params, eps, batchsize)
        self._gamma =gamma

    def run(self, x0, n_epochs, batchsize=None, verbose=20):
        """
        Run stochastic gradient decent of n_epochs epochs

        @returns the samples and energies after each epoch
        """
        if batchsize is None:
            batchsize = self._batchsize

        energies = []
        llambda = 0
        n_data = len(self._data)
        xs = [x0.copy()]
        x = x0.copy()
        old_grad = np.zeros_like(x)
        v =  np.zeros_like(x)
        energies = []

        # Initial update 
        self.update_rotations(x, sample_params=False)

        for epoch in xrange(n_epochs):
            batches = chunks(np.random.permutation(n_data), batchsize)

            for batch in batches:
                grad =  self._batch_gradient(x - self._gamma * v, batch)
                v = self._gamma * v + self._initial_learning_rate * grad
                x -= v
            xs.append(x)
            energies.append(self.energy(x))

            self.update_rotations(x, sample_params=False)
            print energies[-1], np.linalg.norm(grad)

        return xs, energies

class GibbsAdam(GibbsSGD):

    def __init__(self,  likelihood, projection, quadrature,
                 data, prior=None, params=None,
                 eps=1e-8, decay_rate=1e-2,
                 batchsize=500,
                 alpha=1e-3, beta1=0.9, beta2=0.9):
        super(GibbsAdam, self).__init__(likelihood, projection, quadrature,
                                        data, prior, params, eps, batchsize)
        self._beta1 = beta1 
        self._beta2 = beta2
        self._eps = eps
        self._alpha = alpha


    def run(self, x0, n_epochs, batchsize=None, verbose=20):
        """
        Run stochastic gradient decent of n_epochs epochs

        @returns the samples and energies after each epoch
        """
        if batchsize is None:
            batchsize = self._batchsize

        energies = []

        n_data = len(self._data)
        xs = [x0.copy()]
        x = x0.copy()
        energies = []
        t0 = self._initial_learning_rate
        t = 0 
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        m_hat = np.zeros_like(x)
        v_hat = np.zeros_like(x)

        # Initial update 
        self.update_rotations(x, sample_params=False)

        for epoch in xrange(n_epochs):
            batches = chunks(np.random.permutation(n_data), batchsize)

            for batch in batches:
                t += 1
                grad =  self._batch_gradient(x, batch)

                m = self._beta1 * m + (1 - self._beta1) * grad
                v = self._beta2 * v + (1 - self._beta2) * grad**2

                m_hat = m/(1- self._beta1**t)
                v_hat = v/(1- self._beta2**t)
                x = x - self._alpha * m_hat / (np.sqrt(v_hat) + self._eps)
                print np.linalg.norm(m_hat / (np.sqrt(v_hat) + self._eps))

            xs.append(x)
            energies.append(self.energy(x))

            self.update_rotations(x, sample_params=False)
            print epoch, energies[-1], np.linalg.norm(grad)
        return xs, energies
