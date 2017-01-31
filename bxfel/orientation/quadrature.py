import numpy as np
import os
import glob

from abc import abstractmethod, abstractproperty

import bxfel

class AbstractSO3Quadrature(object):

    def __init__(self, order=1):
        self.n = order
        mod_path = os.path.dirname(bxfel.__file__)
        self._path = os.path.join(mod_path,
                                  "orientation/resources")
        self._R = None
        self._w = None

        self._read_file()

    @abstractmethod
    def _read_file(self):
        pass

    @property
    def R(self):
        return self._R

    @property
    def w(self):
        return self._w

    @abstractmethod
    def get_support_rotations(self):
        """
        Returns the support rotations
        """
        pass

    @abstractmethod
    def get_weights(self):
        """
        Returns the weight of the support rotations
        """
        pass


class GaussSO3Quadrature(AbstractSO3Quadrature):

    def __init__(self, order=1):
        super(GaussSO3Quadrature,self).__init__(order)

    def _read_file(self):
        n = self.n
        path = os.path.join(self._path, "gauss")
        files = glob.glob(path + "/*.dat")
        prefix = "N{0:0>2}".format(n)

        file_found = False
 
        for fn in files:
            if prefix in fn:
                file_found = True
                break
        if not file_found:
            raise ValueError("No File to match prefix {}".format(prefix))
                
        with open(fn) as f:
            lines = f.readlines()[2:]

        w = []
        R = []
        for line in lines:
            values = map(float, line.split())
            R.append(np.array(values[:9]).reshape((3, 3)))
            w.append(values[-1])

        self._w = np.array(w)
        self._R = np.array(R)
        self.m = len(w)


class ChebyshevSO3Quadrature(AbstractSO3Quadrature):

    def __init__(self, order=1):
        super(ChebyshevSO3Quadrature,self).__init__(order)
        
    def _read_file(self):
        n = self.n
        path = os.path.join(self._path, "chebyshev")
        files = glob.glob(path + "/*.dat")
        prefix = "N{0:0>2}".format(n)
        file_found = False
 
        for fn in files:
            if prefix in fn:
                file_found = True
                break
        if not file_found:
            raise ValueError("No File to match prefix {}".format(prefix))
        self._fn = fn

        self._fn = fn
        with open(fn) as f:
            lines = f.readlines()[2:]

        R = []
        for line in lines:
            values = map(float, line.split())
            R.append(np.array(values[:9]).reshape((3, 3)))

        self._R = np.array(R)
        self.m = len(R)
        self._w = np.ones(self.m)




if __name__ == "__main__":
    g = GaussSO3Quadrature(14)
    g = ChebyshevSO3Quadrature(6)
