import numpy as np
import unittest
import os

from bxfel.core.structure_factor import ScatteringFactor
from csb.bio.io import StructureParser

class TestScatteringFactor(unittest.TestCase):

    def setUp(self):
        structure = StructureParser(os.path.expanduser("~/projects/xfel/data/GTT_short.pdb")).parse()
        self.f = ScatteringFactor(structure)

        self._n = n = 21
        r_min, r_max = -1, 1
        h, k = np.meshgrid(np.linspace(r_min, r_max,n),
                           np.linspace(r_min, r_max,n))
        l = np.zeros(h.shape)
        self._hkl = np.vstack([item.ravel() for item in [h,k,l]]).T

    def testStructureFactor(self):
        sf = self.f
        hkl = self._hkl
        X = np.array([a.vector for a in  sf._atoms])
        sf.calculate_structure_factors(X,hkl)


    def testGradient(self):
        sf = self.f
        hkl = self._hkl
        X = np.array([a.vector for a in  sf._atoms])
        dg, g = sf.grad_hkl(hkl, X)

        f = sf._calculate_scattering_factors( hkl, X)
        f += sf._calculate_debyewaller_factors(hkl)

        for ff,gg in zip(f.ravel(),g.ravel()):
            self.assertAlmostEqual(ff,gg)

if __name__ == "__main__":
    unittest.main()
