import numpy as np
import copy

import ctypes
from threading import Thread, Lock
from Queue import Queue
from multiprocessing import Pool
import multiprocessing as mp
from  multiprocessing import sharedctypes

from csbplus.mcmc import HMC

class Sampler(object):

    """
    This class implements a Gibbs sampling approach where we iterate between
    sampling new orientations conditioned on the current volume and a new volume
    conditioned on the current orientations
    """


    def __init__(self, posterior):
        self.posterior = posterior
        self.samples = []


    def run(self, x0, niter, eps):
        x = copy.copy(x0)
        hmc = HMC(potential_energy=self.posterior,
                  nsteps=20, eps=eps)

        hmc._adaptphase = 1.
        self.posterior.sample_rotations(x)

        for i in range(niter):
            # Sample new Volume
            s = hmc.run(x, 5, return_momenta=False)
            x = s[-1]

            # Sample new Volume
            self.samples.append(x)
            self.posterior.sample_rotations(x)
            self.posterior.sample_nuissance_parameters(x)

        return self.samples


if __name__ == "__main__":
    import numpy as np
    import scipy as sp

    import os
    import sys

    from bxfel.core.create_data import GaussianSlices, PoissonSlices

    from bxfel.io import mrc
    from bxfel.orientation.quadrature import GaussSO3Quadrature, ChebyshevSO3Quadrature
    from bxfel.model.likelihood  import GaussianLikelihood
    from bxfel.model.posterior import FullPosterior

    from bxfel.orientation.interpolation_matrix import compute_slice_interpolation_matrix, get_image_to_sparse_projection
    from bxfel.core.structure_factor import ScatteringFactor

    import scipy.ndimage
    import time
    from csb.bio.io.wwpdb import get
    if False:
        pdb_id = '4AKE'
        structure = get(pdb_id)
        from csb.bio.structure import Structure
        scale = 1e-4
        resolution = 71
        sf = ScatteringFactor(Structure.from_chain(structure.first_chain))
        x = np.linspace(-0.2, 0.2, resolution)
        h, k, l = np.meshgrid(x,x,x)
        hkl = np.vstack([item.ravel()
                         for item in [h,k,l]]).T
        X = np.array([a.vector for a in  sf._atoms])
        X -= X.mean(0)
        F = sf.calculate_structure_factors(X,hkl)
        F  =  F.reshape((resolution,resolution,resolution))
        ground_truth = I =  (F * np.conj(F)).real
        rho = np.fft.fftshift(np.abs(np.fft.ifftn(F, [250,250,250])))

        from csb.io import load, dump
        mrc.write(I, os.path.expanduser("~/{}_intensity.mrc".format(pdb_id)))
        mrc.write(rho, os.path.expanduser("~/{}_rho.mrc".format(pdb_id)))
        gs = PoissonSlices(ground_truth, scale = scale)
        rs, data  = gs.generate_data(2048, resolution)
        dump(data, os.path.expanduser("~/{}_data_poisson.kl".format(pdb_id)))
        gs = GaussianSlices(ground_truth, scale = scale, sigma=1.)
        rs, data  = gs.generate_data(2048, resolution)
        dump(data, os.path.expanduser("~/{}_data_gaussian.pkl".format(pdb_id)))

    from csb.io import dump, load


    rad = 0.95
    q = ChebyshevSO3Quadrature(15)
    m = len(q.R)
    resolution = 71
    ground_truth = mrc.read(os.path.expanduser("~/{}_intensity.mrc".format("1AKE")))[0]
    d1 = load(os.path.expanduser("~/1AKE_data_gaussian.pkl"))
    d2 = load(os.path.expanduser("~/4AKE_data_gaussian.pkl"))
    data = np.vstack((d1,d2))
    proj = compute_slice_interpolation_matrix(q.R, resolution, radius_cutoff=rad)

    image_to_vector = get_image_to_sparse_projection(resolution, rad)

    d_sparse = np.array([image_to_vector.dot(data[i,:,:].ravel())
                          for i in range(data.shape[0])])

    premult = None

    ll = GaussianLikelihood(1.)
    posterior = FullPosterior(ll, proj, q, d_sparse)
    posterior._likelihood._sample_gamma = True

    opt = Sampler(posterior)
    
    samples = opt.run(x0 + 10* np.random.normal(size=x0.size),
                      25,
                      eps = 1e-4)
    dump(samples, os.path.expanduser("gibbs_results.pkl"))
