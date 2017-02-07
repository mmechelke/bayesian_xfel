import unittest

import numpy as np
from bxfel.orientation.exponential_coordinates import ExponentialCoordinates

class TestExponentialCoordinates(unittest.TestCase):

    def testGradient(self):
        er = ExponentialCoordinates.from_vector(np.random.random(3,))
        dR = er.gradient()
        dR_num = dR * 0.0
        eps = 1e-5
        R0 = er.get_rotation_matrix()
        v = er.v
        for i in range(3):
            v[i] += eps
            R_prime = ExponentialCoordinates.from_vector(v).get_rotation_matrix()
            dR_num[i,:,:] = ((R_prime - R0)/eps)
            v[i] -= eps

        for a,b in zip(dR.ravel(),dR_num.ravel()):
            self.assertAlmostEqual(a,b,delta=1e-4)

    def testMulGradient(self):
        u = np.random.random(3,)

        er = ExponentialCoordinates.from_vector(np.random.random(3,))
        R = er.get_rotation_matrix()
        v = er.v

        Ru0 = np.dot(R, u)
        eps = 1e-5

        g = er.mult_gradient(u)

        g_num = g * 0.
        for i in range(3):
            v[i] += eps
            R_prime = ExponentialCoordinates.from_vector(v).get_rotation_matrix()
            Ru_prime = np.dot(R_prime,u)
            g_num[:,i] = ((Ru_prime - Ru0)/eps)
            v[i] -= eps

        for a,b in zip(g.ravel(),g_num.ravel()):
            self.assertAlmostEqual(a,b,delta=1e-4)

if __name__ == "__main__":
    unittest.main()
