import os

import unittest


import numpy as np
from scipy.ndimage import zoom

from bxfel.model.interpolation_matrix import _get_coords, compute_interpolation_matrix, compute_slice_interpolation_matrix
from bxfel.io import mrc

class InterpolationMatrixTest(unittest.TestCase):


    def test_get_coords_2d(self):

        n_pixel = 20
        coords = _get_coords(n_pixel, ndim=2)
        self.assertEqual(coords.shape[0], n_pixel * n_pixel)
        self.assertEqual(coords.shape[1], 3)

        self.assertEqual(coords[0,0], -10)
        self.assertEqual(coords[-1,0], 9)
        self.assertTrue(np.all(coords<=n_pixel/2))

        n_pixel = 21
        coords = _get_coords(n_pixel, ndim=2)
        self.assertEqual(coords.shape[0], n_pixel * n_pixel)
        self.assertEqual(coords.shape[1], 3)

        self.assertEqual(coords[0,0], -10)
        self.assertEqual(coords[-1,0], 10)

        self.assertTrue(np.all(coords<=n_pixel/2))

    def test_get_coords_3d(self):
        n_pixel = 20
        coords = _get_coords(n_pixel, ndim=3)
        self.assertEqual(coords.shape[0], n_pixel * n_pixel * n_pixel)
        self.assertEqual(coords.shape[1], 3)

        self.assertEqual(coords[0,0], -10)
        self.assertEqual(coords[-1,0], 9)
        self.assertTrue(np.all(coords<=n_pixel/2))

        n_pixel = 21
        coords = _get_coords(n_pixel, ndim=3)
        self.assertEqual(coords.shape[0], n_pixel * n_pixel * n_pixel)
        self.assertEqual(coords.shape[1], 3)

        self.assertEqual(coords[0,0], -10)
        self.assertEqual(coords[-1,0], 10)
        self.assertTrue(np.all(coords<=n_pixel/2))

    def test_get_coords_2d_cutoff(self):

        n_pixel = 21
        cutoff = np.random.random()
        coords = _get_coords(n_pixel,radial_cutoff=cutoff, ndim=2)
        self.assertLessEqual(coords.shape[0], n_pixel * n_pixel)
        self.assertEqual(coords.shape[1], 3)

        dist = np.array(map(np.linalg.norm, coords))
        self.assertTrue(np.all(dist<= n_pixel * cutoff/2))
        
    def test_get_coords_3d_cutoff(self):
        n_pixel = 21
        cutoff = np.random.random()
        coords = _get_coords(n_pixel,radial_cutoff=cutoff, ndim=3)
        self.assertLessEqual(coords.shape[0], n_pixel * n_pixel * n_pixel)
        self.assertEqual(coords.shape[1], 3)

        dist = np.array(map(np.linalg.norm, coords))
        self.assertTrue(np.all(dist<= n_pixel * cutoff/2))

    def test_interpolation_matrix(self):
        density = mrc.read(os.path.expanduser("~/projects/xfel/data/phantom/phantom.mrc"))[0]
        n_voxel = 21
        density = zoom(density, n_voxel/128.)

        twoD = density[:,:,10]

        target = [np.eye(3),]
        proj = compute_slice_interpolation_matrix(target, n_voxel,
                                                  radius_cutoff=None)

        twoD_proj = proj.dot(density.ravel())

        self.assertEqual(twoD_proj.size,  twoD.size)
        for a,b in zip(twoD_proj.ravel(), twoD.ravel()):
            self.assertAlmostEqual(a,b)


    def test_interpolation_matrix_n_pixel(self):
        gt = mrc.read(os.path.expanduser("~/projects/xfel/data/phantom/phantom.mrc"))[0]
        n_voxel = 31
        n_pixel = 21
        density = zoom(gt, n_voxel/128.)
        twoD = zoom(gt, n_pixel/128.)[:,:,10]

        target = [np.eye(3),]
        proj = compute_slice_interpolation_matrix(target, n_voxel, n_pixel,
                                                  radius_cutoff=None)

        twoD_proj = proj.dot(density.ravel())

        self.assertEqual(twoD_proj.size,  twoD.size)
        self.assertTrue(np.corrcoef(twoD_proj.ravel(), twoD.ravel())[0,1] > 0.99)


if __name__ == "__main__":

    unittest.main()
