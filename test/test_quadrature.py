import unittest
from bxfel.orientation.quadrature import GaussSO3Quadrature, ChebyshevSO3Quadrature
import numpy as np

class TestGauss(unittest.TestCase):

    def test_init(self):

        g = GaussSO3Quadrature(1)

        self.assertEqual(g.m, 4)
        self.assertTrue(g._R is not None)
        self.assertTrue(g._w is not None)
        self.assertEqual(g._w[0], 0.25)
        self.assertTrue(np.array_equal(g._R[0].ravel(),
                                       np.eye(3).ravel()))

    def test_Rotation(self):

        i = np.random.randint(1,10)
        g = GaussSO3Quadrature(i)

        for R in g._R:
            self.assertAlmostEqual(np.linalg.det(R), 1.)


class TestGauss(unittest.TestCase):

    def test_init(self):

        g = GaussSO3Quadrature(1)

        self.assertEqual(g.m, 4)
        self.assertTrue(g._R is not None)
        self.assertTrue(g._w is not None)
        self.assertEqual(g._w[0], 0.25)
        self.assertTrue(np.array_equal(g._R[0].ravel(),
                                       np.eye(3).ravel()))

    def test_Rotation(self):

        i = np.random.randint(1,10)
        g = GaussSO3Quadrature(i)

        for R in g._R:
            self.assertAlmostEqual(np.linalg.det(R), 1.)


        
if __name__ == "__main__":
    unittest.main()
