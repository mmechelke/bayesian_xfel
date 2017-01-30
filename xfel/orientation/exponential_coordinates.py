
import numpy as np


def skew3(v):
    a = np.zeros((3, 3))
    a[0,1] = -v[2]
    a[0,2] = v[1]
    a[1,0] = v[2]
    a[1,2] = -v[0]

    a[2,0] = -v[1]
    a[2,1] = v[0]
    return a


class ExponentialCoordinates(object):

    def __init__(self, v, theta):
        self._v = v
        self._theta = theta


    @property
    def vbar(self):
        return self._v


    @property
    def v(self):
        return self._v * self._theta

    @v.setter
    def v(self,value):
        theta  = np.linalg.norm(v)
        if theta > np.pi:
            raise ValueError("Theta: {} > pi, rotation not defined".format(theta))
        self._theta = theta
        self._v = value/self._theta

    @property
    def theta(self):
        return self._theta

    def get_rotation_matrix(self):
        v = self._v
        if np.linalg.norm(v) == 0.:
            return np.eye(3)
        theta = self._theta
        mv = skew3(v)
        R = np.cos(theta) * np.eye(3) + np.sin(theta)\
            * mv + (1 - np.cos(theta)) * np.multiply.outer(v, v)
        return R

    @classmethod
    def from_matrix(cls, R):
        if np.allclose(R,np.eye(3)):
            return cls(np.zeros((3,)), 0.0)

        theta = np.arccos(0.5 * (np.trace(R) - 1.))
        v = np.zeros((3,))
        v[0] = R[2,1] - R[1,2]
        v[1] = R[0,2] - R[2,0]
        v[2] = R[1,0] - R[0,1]
        v /= 2 * np.sin(theta)
        return cls(v, theta)

    @classmethod
    def from_vector(cls, v):
        theta  = np.linalg.norm(v)
        if theta < 1e-109:
            return cls(v, theta)
        return cls(v/theta, theta)

    def gradient(self):
        v = self.v
        R = self.get_rotation_matrix()
        grad_v = np.zeros((3,3,3))
        for i in range(3):
            a = v[i] * skew3(v)
            b = np.cross(v, np.dot(np.eye(3) - R, np.eye(3)[i]))
            b =skew3(b)
            grad_v[i] = np.dot((a + b)/np.linalg.norm(v)**2,R)
        return grad_v

    def mult_gradient(self,u):
        v = self.v
        R = self.get_rotation_matrix()

        a = np.dot(-R, skew3(u))
        b = np.multiply.outer(v, v) + np.dot(R.T - np.eye(3), skew3(v))
        b /= np.linalg.norm(v)**2
        return np.dot(a, b)

if __name__ == "__main__":
    from csbplus.statistics.rand import uniform_random_rotation

    ex = ExponentialCoordinates(np.eye(3)[0], np.pi)
    R = ex.get_rotation_matrix()
    er = ExponentialCoordinates.from_matrix(R)

    R_opt = uniform_random_rotation(1)
    er = ExponentialCoordinates.from_matrix(R_opt.T)

    Y = np.random.random((10,3))
    X = np.dot(Y,R_opt)

    def energy(v, X, Y):
        R = ExponentialCoordinates.from_vector(v).get_rotation_matrix()
        return 0.5 * np.sum((Y-np.dot(R,X.T).T)**2)

    def gradient(v, X, Y):
        ec = ExponentialCoordinates.from_vector(v)
        R = ec.get_rotation_matrix()
        diff = Y - np.dot(R,X.T).T
        g = ec.gradient()
        grad = np.zeros(3)
        for i in range(3):
            grad[i] = np.sum(diff * np.dot(-X,g[i].T))
        return grad


    import scipy.optimize as opt
    x0 = opt.minimize(energy, x0 = np.random.random(3,), args=(X,Y),jac=gradient)
    print "Gradient: ", x0.fun, x0.x
    x0 = opt.minimize(energy, x0 = np.random.random(3,), args=(X,Y))
    print "Approx: ", x0.fun, x0.x
    print energy(er.v,X,Y)


    dx = np.array([np.dot(X[0].T,dR_dv[i],) for i in range(3)])

    dx_dv = er.mult_gradient(X[0])

    er2 = ExponentialCoordinates(er._v[:], er._theta)
    er2._v[-1] = 2.

    e0 =  energy(er2.v,X,Y)
    grad = np.zeros(3)
    grad2 = gradient(er2.v,X,Y)
    eps = 1e-8
    for i in range(3):
        er2._v[i] += eps
        grad[i] = (energy(er2.v,X,Y) - e0)/eps
        er2._v[i] -= eps

    print grad,grad2
