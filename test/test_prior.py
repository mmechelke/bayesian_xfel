import unittest
import numpy as np
from abc import ABCMeta


from bxfel.model.prior import LaplacePrior, DoubleExpPrior, LocalMeanPrior

class AbstractFiniteDiffTest(object):
    __metaclass__ = ABCMeta


    def test_finite_difference(self):
        x = np.random.normal(size=self._size).ravel()

        energy = self.prior.energy(x)
        grad = self.prior.gradient(x)
        num_grad = np.zeros(grad.shape)

        for i in range(self._n):
            x[i] += self._eps
            num_grad[i] = (self.prior.energy(x) - energy)/self._eps
            x[i] -= self._eps

        np.testing.assert_array_almost_equal(grad, num_grad, 2)


class TestLaplacePrior(AbstractFiniteDiffTest, unittest.TestCase):

    def setUp(self):
        self.prior = LaplacePrior()
        self._sampler = np.random.normal
        self._n = 1000
        self._eps = 1e-7
        self._size = (10,10,10)

class TestDoubleExpPrior(AbstractFiniteDiffTest, unittest.TestCase):

    def setUp(self):
        self.prior = DoubleExpPrior(30., 0.1)
        self._sampler = np.random.normal
        self._n = 1000
        self._eps = 1e-7
        self._size = (10,10,10)



class TestLocalPrior(AbstractFiniteDiffTest, unittest.TestCase):

    def setUp(self):
        self.prior = LocalMeanPrior(2., 10)
        self._sampler = np.random.normal
        self._n = 1000
        self._eps = 1e-7
        self._size = (10,10,10)


if __name__ == "__main__":
    unittest.main()


