import os

import numpy as np
import unittest
import bxfel

from bxfel.core.create_data import GaussianSlices, PoissonSlices
from bxfel.io import mrc

class TestPoissonSlices(unittest.TestCase):

    def setUp(self):

        data_path = os.path.dirname(bxfel.__file__)
        phantom_path = os.path.join(data_path,
                                    "data/volumes/phantom.mrc")

        self.volume =  mrc.read(phantom_path)[0]
        self.n = self.volume.shape[0]
        self.generator = PoissonSlices(self.volume)
        scale = 1e7

    def test_generation(self):
        n_samples = int(1e2)
        resolution = self.volume.shape[0]
        rs, data = self.generator.generate_data(n_samples, resolution)

        self.assertEqual(rs.shape[0], n_samples)
        self.assertEqual(rs.shape[1], 3)
        self.assertEqual(rs.shape[2], 3)

        self.assertEqual(data.shape[1], resolution)
        self.assertEqual(data.shape[2], resolution)

        for i in range(n_samples):
            self.assertAlmostEqual(np.linalg.det(rs[i]),1.)
        self.assertEqual(data.shape[0], n_samples)

        n_samples = int(1e2)
        resolution = np.random.randint(21,self.volume.shape[0])
        rs, data = self.generator.generate_data(n_samples, resolution)

        self.assertEqual(rs.shape[0], n_samples)
        self.assertEqual(rs.shape[1], 3)
        self.assertEqual(rs.shape[2], 3)

        self.assertEqual(data.shape[1], resolution)
        self.assertEqual(data.shape[2], resolution)
        for i in range(n_samples):
            self.assertAlmostEqual(np.linalg.det(rs[i]),1.)
        self.assertEqual(data.shape[0], n_samples)


if __name__ == "__main__":
    unittest.main()





