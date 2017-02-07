import numpy as np

from csbplus.statistics.rand import uniform_random_rotation

from bxfel.core.grid import Grid

import scipy.ndimage

class Experiment(object):

    def __init__(self, density):
        n = np.shape(density)[0]

        coords = np.linspace(-1.,1.,n)
        self._n = n
        self._rho = Grid(n,n,n,density)
        self._rho.origin = (n/2., n/2., n/2.)

    def generate_data(self, n_data):
        raise NotImplementedError("Subclass responsability")


class GaussianSlices(Experiment):


    def __init__(self, density, scale, sigma, const_fluence=True, fluence_alpha=10., fluence_beta=10.):
        super(GaussianSlices, self).__init__(density)
        self._scale = float(scale)
        self._sigma = float(sigma)

        self._const_fluence = const_fluence
        self._fluence_alpha = fluence_alpha
        self._fluence_beta = fluence_beta

    def generate_data(self, n_data, resolution=32):
        """
        @param n_data: number of images to generate_data
        @param resolution: size of the grid
        """
        samples = []
        Rs = uniform_random_rotation(n_data)
        for R in Rs:
            slice = self._rho.slice(R.T).T
            slice = scipy.ndimage.zoom(slice,
                                       float(resolution)/float(np.shape(slice)[0]))
            scale = self._scale
            if not self._const_fluence:
                scale *= np.random.gamma(self._fluence_alpha, 1./self._fluence_beta)
            noisy_slice = np.random.normal(scale * slice, self._sigma)
            samples.append(noisy_slice)
        return Rs, np.array(samples)

class PoissonGaussianSlices(GaussianSlices):


    def generate_data(self, n_data, resolution=32):
        """
        @param n_data: number of images to generate_data
        @param resolution: size of the grid
        """
        samples = []
        Rs = uniform_random_rotation(n_data)
        for R in Rs:
            slice = self._rho.slice(R.T).T
            slice = scipy.ndimage.zoom(slice,
                                       float(resolution)/float(np.shape(slice)[0]))
            scale = self._scale
            if not self._const_fluence:
                scale *= np.random.gamma(self._fluence_alpha, 1./self._fluence_beta)
            noisy_slice = np.random.normal(scale * slice, self._sigma * np.sqrt(scale* slice))
            samples.append(noisy_slice)
        return Rs, np.array(samples)



class PoissonSlices(Experiment):


    def __init__(self, density, scale = 1.):
        super(PoissonSlices, self).__init__(density)
        self._scale = scale

    def generate_data(self, n_data, resolution=32):
        """
        @param n_data: number of images to generate_data
        @param resolution: size of the grid
        """
        samples = []
        Rs = uniform_random_rotation(n_data)
        for R in Rs:
            slice = self._rho.slice(R.T).T
            slice = slice.clip(0.,1e300)
            slice = scipy.ndimage.zoom(slice,
                                       float(resolution)/float(np.shape(slice)[0]))
            noisy_slice = np.random.poisson(self._scale * slice.clip(0.,1e100))
            samples.append(noisy_slice)
        return Rs, np.array(samples)


