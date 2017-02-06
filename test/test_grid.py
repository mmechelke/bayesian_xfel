import unittest
import numpy as np

import os
import bxfel

from bxfel.io import mrc
from bxfel.core.grid import Grid

class TestGrid(unittest.TestCase):

    def setUp(self):

        self.grid = Grid(50,50,50)

        data_path = os.path.dirname(bxfel.__file__)
        phantom_path = os.path.join(data_path,
                                    "data/volumes/phantom.mrc")

        volume =  mrc.read(phantom_path)[0]
        n = volume.shape[0]
        self.grid_with_data = Grid(n,n,n, volume)


    def testInit(self):
        shape = self.grid.shape
        self.assertEqual(shape[0], 50)
        self.assertEqual(shape[1], 50)
        self.assertEqual(shape[2], 50)

        spacing = self.grid.spacing
        self.assertEqual(spacing, 1.)

        width = self.grid.width
        self.assertEqual(width, 1.)

        origin = self.grid.origin
        self.assertEqual(origin[0], 0.0)
        self.assertEqual(origin[1], 0.0)
        self.assertEqual(origin[2], 0.0)

        self.assertIsNotNone(self.grid.values)

        grid_shape = len(self.grid.values)
        self.assertEqual(shape[0] * shape[1] * shape[2], grid_shape)


    def testSetDensity(self):
        a = self.grid
        a.set_density(0.5)
        self.assertEqual(a.values[0], 0.5)
        a.set_density(0.25)
        self.assertEqual(a.values[0], 0.25)


    def testClone(self):

        a = self.grid
        a.set_density(0.5)
        self.assertEqual(a.values[0], 0.5)
        b = a.clone()
        self.assertEqual(b.values[0], 0.5)

        a.set_density(0.0)
        self.assertEqual(b.values[0], 0.5)
        self.assertEqual(a.values[0], 0.0)

        a = self.grid_with_data
        b = a.clone()

        dataa = a.get_data()
        datab = b.get_data()

        assert np.array_equal(dataa, datab)


    def test_transform_and_interpolate(self):
        a = self.grid
        R = np.eye(3)
        t = np.zeros(3)
        g = self.grid_with_data 
        g2 = g.transform(R,t)

        dataa = g.get_data()
        datab = g2.get_data()

        np.testing.assert_array_equal(dataa, datab)

    def test_slice(self):

        g = self.grid_with_data
        nz = g.nz
        idz = int((nz+1)/2)

        s = g.slice(np.eye(3))
        st = g.get_data()[:,:,idz]

        self.assertEqual(s.shape[0], g.nx)
        self.assertEqual(s.shape[1], g.ny)

        np.testing.assert_array_almost_equal(s, st)






if __name__ == '__main__':
    unittest.main()
    
