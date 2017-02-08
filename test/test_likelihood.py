import unittest
import os, sys
import numpy as np
from functools import partial

from bxfel.io import mrc

from bxfel.model.likelihood import GaussianLikelihood, PoissonLikelihood, TiedGaussianLikelihood, AnscombeLikelihood
from abc import ABCMeta



class AbstractFiniteDiffTest(object):
    __metaclass__ = ABCMeta

    def setUp(self):
        self._ll = None
        self._sampler = np.random.normal
        self._n = 1000
        self._eps = 1e-7

    def test_energy(self):
        pass

    def test_finite_difference(self):
        x = np.random.gamma(10,10, size=self._n)
        d = np.random.gamma(10,10, size=self._n)

        energy = self._ll.energy(x,d)
        _, grad = self._ll.gradient(x,d)
        num_grad = np.zeros(grad.shape)

        for i in range(self._n):
            x[i] += self._eps
            num_grad[i] = (self._ll.energy(x,d) - energy)/self._eps
            x[i] -= self._eps

        np.testing.assert_array_almost_equal(grad, num_grad, 2)

class TestGaussian(AbstractFiniteDiffTest, unittest.TestCase):

    def setUp(self):
        k = 0.5
        self._sampler = np.random.normal
        self._n = 1000

        self._ll = GaussianLikelihood(k, self._n)
        self._eps = 1e-7


    def test_sample_force_constant(self):
        n = self._n
        self._ll._sample_gamma = False
        self._ll._sample_k = True
        params = {"k":0.1, "gamma":1.}
        sigma = 1.
        data = np.random.normal(scale=sigma, size=n)
        self._ll.sample_nuissance_params(np.random.normal(data), data, params)
        self.assertTrue(params["k"] != 0.1)
        ks = []
        for i in range(100):
            self._ll.sample_nuissance_params(np.random.normal(data), data, params)
            ks.append(params["k"])

        self.assertTrue(np.abs(np.mean(ks) -  1.) < 0.1)

    def test_sample_gamma(self):
        n = self._n

        self._ll._sample_gamma = True
        self._ll._sample_k = False

        params = {"gamma":1., "k":1.}
        data = np.random.normal(1., scale=1., size=n)
        self._ll.sample_nuissance_params(np.random.normal(data), data, params)
        self.assertTrue(params["gamma"] != 1.)

        gammas = []
        for i in range(100):
            self._ll.sample_nuissance_params(2 * (data), data, params)
            gammas.append(params["gamma"])
        self.assertTrue(np.abs(np.mean(gammas) -  0.5) < 0.1)


class TestPoisson(AbstractFiniteDiffTest, unittest.TestCase):

    def setUp(self):
        self._ll = PoissonLikelihood()
        self._sampler = np.random.poisson
        self._n = 1000
        self._eps = 1e-5

class TestAnscombe(AbstractFiniteDiffTest, unittest.TestCase):

    def setUp(self):
        self._ll = AnscombeLikelihood()
        self._sampler = np.random.poisson
        self._n = 1000
        self._eps = 1e-5


class TestTied(AbstractFiniteDiffTest, unittest.TestCase):

    def setUp(self):
        self._ll = TiedGaussianLikelihood()
        self._sampler = np.random.poisson
        self._n = 1000
        self._eps = 1e-5



if __name__ == "__main__":
    unittest.main()
