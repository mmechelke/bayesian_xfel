import numpy as np
import os, sys

import unittest

from bxfel.io import mrc
from abc import ABCMeta

from bxfel.model.posterior import IntegratedOrientationPosterior, FullPosterior
from bxfel.model.likelihood import GaussianLikelihood
from bxfel.orientation.quadrature import GaussSO3Quadrature, ChebyshevSO3Quadrature
from bxfel.core.create_data import GaussianSlices

from bxfel.model.interpolation_matrix import  get_image_to_sparse_projection, compute_interpolation_matrix, compute_slice_interpolation_matrix

class AbstractPosteriorTest(object):
    __metaclass__ = ABCMeta

    def gen_model(self):
        from scipy.ndimage import zoom
        rad = 0.95
        n_data = 100
        self.n_data = n_data
        self._eps = 1e-5
        self._resolution = resolution = 5

        ground_truth = mrc.read(os.path.expanduser("~/projects/xfel/data/phantom/phantom.mrc"))[0]
        ground_truth = zoom(ground_truth, resolution/128.)

        q = ChebyshevSO3Quadrature(5)

        gs = GaussianSlices(ground_truth, scale = 1., sigma = 2.)
        rs, data  = gs.generate_data(n_data, resolution)

        proj = compute_slice_interpolation_matrix(q.R, resolution,
                                                  radius_cutoff=rad)

        image_to_vector = get_image_to_sparse_projection(resolution, rad)

        d_sparse = np.array([image_to_vector.dot(data[i,:,:].ravel())
                             for i in range(data.shape[0])])

        ll = GaussianLikelihood(1.)
        return ll, proj, q, d_sparse

    def gen_model_and_masks(self):
        from scipy.ndimage import zoom
        rad = 0.95
        n_data = 100
        self._ndata = n_data
        self._eps = 1e-6
        self._resolution = resolution = 5

        ground_truth = mrc.read(os.path.expanduser("~/projects/xfel/data/phantom/phantom.mrc"))[0]
        ground_truth = zoom(ground_truth, resolution/128.)

        q = ChebyshevSO3Quadrature(5)

        gs = GaussianSlices(ground_truth, scale = 1., sigma = 2.)
        rs, data  = gs.generate_data(n_data, resolution)

        masks = np.zeros_like(data)
        masks[:,2,2] = 1

        proj = compute_slice_interpolation_matrix(q.R, resolution,
                                                  radius_cutoff=rad)
        image_to_vector = get_image_to_sparse_projection(resolution, rad)

        d_sparse = np.array([image_to_vector.dot(data[i,:,:].ravel())
                             for i in range(data.shape[0])])

        m_sparse = np.array([image_to_vector.dot(masks[i,:,:].ravel())
                             for i in range(data.shape[0])])

        self._masks = m_sparse
        ll = GaussianLikelihood(1.)
        return ll, proj, q, d_sparse, m_sparse


    def test_finite_difference(self):
        x = np.random.random(self._resolution**3)
        grad = self._posterior.gradient(x)
        energy = self._posterior.energy(x)
        num_grad = np.zeros(grad.shape)

        for i in range(x.size):
            x[i] += self._eps
            num_grad[i] = (self._posterior.energy(x) - energy)/self._eps
            x[i] -= self._eps

        np.testing.assert_array_almost_equal(grad, num_grad,1)


class IntegratedOrientationPosteriorTest(AbstractPosteriorTest, unittest.TestCase):

    def setUp(self):
        ll, proj, q, d_sparse = self.gen_model()
        self._posterior = IntegratedOrientationPosterior(ll, proj, q, d_sparse)

class FullPosteriorTest(AbstractPosteriorTest, unittest.TestCase):

    def setUp(self):
        ll, proj, q, d_sparse = self.gen_model()
        self._posterior = FullPosterior(ll, proj, q, d_sparse)


if __name__ == "__main__":
    unittest.main()
