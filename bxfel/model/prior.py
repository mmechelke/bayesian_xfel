import numpy as np

from abc import ABCMeta, abstractmethod

class Prior(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def energy(self, x):
        raise NotImplementedError("Not implemented in Base class")

    @abstractmethod
    def gradient(self, x):
        raise NotImplementedError("Not implemented in Base class")



class LaplacePrior(Prior):

    def __init__(self, k=1.):
        self._k = k

    def energy(self, x):
        return self._k * np.sum(np.abs(x))

    def gradient(self, x):
        return self._k * np.sign(x)


class DoubleExpPrior(Prior):

    """
    Double exponential prior with two parameters
    controlling the rate for positive and negative
    values independently
    """

    def __init__(self, lambda_neg, lambda_pos):
        self._lambda_neg = lambda_neg
        self._lambda_pos = lambda_pos

    def energy(self, x):
        ln = self._lambda_neg
        lp = self._lambda_pos
        x_pos = x[x>0]
        x_neg = x[x<0]

        u_pos = -len(x_pos) * np.log(lp)  +  lp * np.sum(x_pos)
        u_neg = -len(x_neg) * np.log(ln)  +  ln * np.sum(-x_neg)

        return u_pos + u_neg

    def gradient(self, x):
        grad = np.zeros_like(x)
        grad[x>0] = self._lambda_pos
        grad[x<0] = -self._lambda_neg

        return grad




class LocalMeanPrior(Prior):

    def __init__(self, k, N):
        """
        assumes that x is in fact
        a N x N x N Tensor
        """
        self._k = k
        self._N = int(N)


    def energy(self, x):
        tmp = x.reshape((self._N, self._N, self._N))
        u = 0.0
        for i in range(self._N):
            for j in range(self._N):
                for k in range(self._N):

                    for l in [-1, 0, 1]:
                        for m in [-1, 0, 1]:
                            for n in [-1, 0, 1]:
                                if l==0 and m==0 and n==0:
                                    continue
                                if (i+l >= 0 and i+l< self._N
                                    and j+m >= 0 and j+m < self._N
                                    and k+n >= 0 and k+n < self._N):
                                    u += 0.5 * self._k * (tmp[i,j,k] - tmp[i+l,j+m,k+n])**2
        return u

    def gradient(self, x):
        tmp = x.reshape((self._N, self._N, self._N))
        grad = np.zeros_like(tmp)

        u = 0.0
        for i in range(self._N):
            for j in range(self._N):
                for k in range(self._N):

                    for l in [-1, 0, 1]:
                        for m in [-1, 0, 1]:
                            for n in [-1, 0, 1]:
                                if l==0 and m==0 and n==0:
                                    continue
                                if (i+l >= 0 and i+l< self._N
                                    and j+m >= 0 and j+m < self._N
                                    and k+n >= 0 and k+n < self._N):
                                    grad[i,j,k] +=  self._k * (tmp[i,j,k] - tmp[i+l,j+m,k+n])
                                    grad[i+l,j+m,k+n] -=  self._k * (tmp[i,j,k] - tmp[i+l,j+m,k+n])
        return grad.ravel()

