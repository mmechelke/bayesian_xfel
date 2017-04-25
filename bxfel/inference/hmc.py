
import ctypes
from threading import Thread, Lock
from Queue import Queue
from multiprocessing import Pool
import multiprocessing as mp
from  multiprocessing import sharedctypes

shared_slices = None
shared_data = None
shared_grad = None

#TODO pass likelihood params

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

    ll = GaussianLikelihood(1.)
    s = shared_slices[slice_index]
    d = shared_data[data_index]
    energy, gradient = ll.gradient(s,d)
    
    return gradient

class GibbsMPsampler(object):

    def __init__(self, posterior):
        """
        @param alpha: prior hyperparameter of the prior

        """
        self._likelihood = posterior._likelihood
        self._projection = posterior._projection
        self._q = posterior._quadrature

        # we need to make sure slices and
        # data are shared arrayes

        self._data = posterior._data
        self._prior = posterior._prior
        self._orientations = np.random.choice(len(self._q.R),
                                              size=len(data))
        if params is None:
            self._params = {}
        else:
            self._params = params

        self.__init__process(n_cpu)

    def __init__process(self, n_cpu):
        """
        Create populate global data
        """
        global shared_slices
        global shared_data

        shared_slices_base = sharedctypes.RawArray(ctypes.c_double,
                                                      self._projection.shape[0])
        shared_slices = np.frombuffer(shared_slices_base)
        shared_slices = shared_slices.reshape((len(self._q.R), -1))

        shared_grad_base = sharedctypes.RawArray(ctypes.c_double,
                                                      self._projection.shape[0])
        shared_grad = np.frombuffer(shared_grad_base)
        shared_grad = shared_grad.reshape((len(self._q.R), -1))

        shared_data_base = mp.Array(ctypes.c_double,
                                    self._data.size,
                                    lock=False)
        shared_data = np.ctypeslib.as_array(shared_data_base)
        shared_data = shared_data.reshape(self._data.shape)
        shared_data[:] = self._data

        self._pool = mp.Pool(n_cpu)

    def energy(self, x):
        # project density to slices
        global shared_slices
        energy = 0

        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        shared_slices[:] = slices[:]

        params = [(self._orientations[i], i)
                  for i in range(len(data))]
        energy_arr = self._pool.map_async(do_energy, params)

        # Maybe we have something to do while we wait 
        if self._prior is not None:
            energy = self._prior.energy(x)
        energy += np.sum(energy_arr.get())

        return energy

    def gradient(self, x):
        # project density to slices
        global shared_slices
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        shared_slices = slices.squeeze()

        params = [(self._orientations[i], i)
                  for i in range(len(data))]
        grad = self._pool.map(do_gradient, params)

        # project back
        backproj_grad = self._projection.T.dot(grad.ravel())
        backproj_grad = np.squeeze(np.asarray(backproj_grad))

        if self._prior is not None:
            backproj_grad += self._prior.gradient(x)
        return backproj_grad

    def __call__(self, x):
        return self.energy(x)


    def update_rotations(self, x, sample_params=False):
        m = len(self._q.R)
        slices = self._projection.dot(x.reshape((-1,)))
        slices = slices.reshape((m, -1))
        slices = np.squeeze(np.asarray(slices))
        shared_slices[:] = slices.squeeze()[:]

        params = [(j, i)
                  for i in range(len(data))
                  for j in range(m)]

        energies = self._pool.map(do_energy, params)
        energies = np.array(energies).reshape((len(data), m))

        p = np.exp(-energies
                   - sp.misc.logsumexp(-energies,-1)[:,np.newaxis])

        for i in range(len(self._data)):
            # For the moment let's trust floating point arithmetic
            # p_i = p[i]/p[i].sum()
            self._orientations[i] = np.random.choice(m, p=p[i])

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

if __name__ == "__main__":
    import numpy as np
    import scipy as sp

    import pylab as plt
    import seaborn

    import os
    import sys

    from xfel.utils.create_data import GaussianSlices

    from xfel.io import mrc
    from xfel.numeric.quadrature import GaussSO3Quadrature, ChebyshevSO3Quadrature
    from xfel.grid.optimize import GaussianLikelihood, Objective, GibbsOrientationSampler


    from xfel.grid.interpolation_matrix import compute_slice_interpolation_matrix, get_image_to_sparse_projection

    import scipy.ndimage
    import time

    from scipy.ndimage import zoom
    ground_truth = mrc.read(os.path.expanduser("~/projects/xfel/data/phantom/phantom.mrc"))[0]
    resolution = 121

    ground_truth = zoom(ground_truth, resolution/128.)

    read_from_disk = True

    from csb.io import load, dump
    if read_from_disk is False:
        gs = GaussianSlices(ground_truth, scale = 1., sigma = 0.01)
        rs, data  = gs.generate_data(2048, resolution)
        dump(data, "/tmp/test_data.pkl")
    else:
        data = load("/tmp/test_data.pkl")

    rad = 0.95
    q = ChebyshevSO3Quadrature(9)
    m = len(q.R)

    if read_from_disk is False:
        proj = compute_slice_interpolation_matrix(q.R, resolution, radius_cutoff=rad)
        dump(proj, "/tmp/test_proj.pkl")
    else:
        proj = load("/tmp/test_proj.pkl")

    image_to_vector = get_image_to_sparse_projection(resolution, rad)

    d_sparse = np.array([image_to_vector.dot(data[i,:,:].ravel())
                          for i in range(data.shape[0])])

    premult = None

    ll = GaussianLikelihood(1.)
    opt = HMCsampler(likelihood=ll,
                    projection=proj,
                    quadrature=q,
                     data=d_sparse,
                     n_cpu = 4)
    t0 = time.time()
    opt.update_rotations(ground_truth.ravel())
    print time.time() - t0
    print "Starting to profile"
    import cProfile
    cProfile.run('opt.update_rotations(ground_truth.ravel())')
    print "Done"



    opt2 = GibbsOrientationSampler(likelihood=ll,
                                   projection=proj,
                                   quadrature=q,
                                   data=d_sparse)


    t0 = time.time()
    opt2.update_rotations(ground_truth.ravel())
    print time.time() - t0

    opt2._orientations = opt._orientations.copy()


    t0 = time.time()
    energy = opt.energy(ground_truth.ravel())
    print time.time() - t0
    t0 = time.time()
    energy2 = opt2.energy(ground_truth.ravel())
    print time.time() - t0

