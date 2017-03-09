import numpy as np

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

        # Now normalize this volume

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
