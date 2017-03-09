import numpy as np
import scipy as sp


from optimize import GibbsOrientationSampler, Likelihood

class CryoEMSampler(GibbsOrientationSampler):


    def energy(self, x):
        fx = np.fft.fftshift(np.fft.fftn(x))
        return super(CryoEMSampler, self).energy(fx)

    def gradient(self, x):
        fx = np.fft.fftshift(np.fft.fftn(x))
        dE =  super(CryoEMSampler, self).gradient(fx)
        return np.fft.ifftshift(np.fft.ifftn(dE))

    def update_rotations(self, x):
        fx = np.fft.fftshift(np.fft.fftn(x))
        return super(CryoEMSampler, self).update_rotations(fx)


class CryoEMGaussianLikelihood(Likelihood):


    def __init__(self, k=1., n=1, mask=None):
        super(CryoEMGaussianLikelihood, self).__init__(n, mask)
        self._k = np.float(k)


    def _energy(self, theta, data):
        n_data = self._n
        chi = (data - theta)
        chi2 = chi.dot(chi)
        E = 0.5 * self._k * chi2
        # Partion function
        E -= n_data * np.log(self._k)
        return E

    def _gradient(self, theta, data):
        n_data = self._n
        diff = (data - theta)
        energy = 0.5 * self._k * diff.dot(diff)
        # Partion function
        energy -=  n_data * np.log(self._k)
        grad = -self._k * diff
        return energy, grad


if __name__ == "__main__":

    import pylab as plt
    import seaborn as sns
    
    import scipy.ndimage
    import time
    from xfel.numeric.quadrature import GaussSO3Quadrature, ChebyshevSO3Quadrature
    import os

    from xfel.grid.interpolation_matrix import compute_slice_interpolation_matrix, get_image_to_sparse_projection
    from xfel.grid.optimize import GaussianLikelihood

    resolution = 32
    order = 3
    rad = 0.99

    q = ChebyshevSO3Quadrature(order)
    m = len(q.R)

    from xfel.io import mrc
    from xfel.grid.interpolation_matrix import compute_slice_interpolation_matrix, get_image_to_sparse_projection

    from scipy.ndimage import zoom
    ground_truth = mrc.read(os.path.expanduser("~/projects/xfel/data/phantom/phantom.mrc"))[0]
    gt = ground_truth = zoom(ground_truth, resolution/128.)

    data = mrc.read(os.path.expanduser("~/projects/gmm-rec/examples/phantom/coarse_input.mrc"))[0]
    data  = data.swapaxes(0,2)
    proj = compute_slice_interpolation_matrix(q.R, resolution, radius_cutoff=rad)

    image_to_vector = get_image_to_sparse_projection(resolution, rad)

    ft_data = np.array([np.fft.fftshift(np.fft.fftn(d)) for d in data])
    ft_data_sparse = np.array([image_to_vector.dot(ft_data[i,:,:].ravel())
                      for i in range(ft_data.shape[0])])

    ll = CryoEMGaussianLikelihood()

    d = gt.sum(-1)
    fd = np.fft.fftshift(np.fft.fftn(d))
    fd = image_to_vector.dot(fd.ravel())
    slices = proj.dot(np.fft.fftshift(np.fft.fftn(gt)).ravel())
    slices = slices.reshape((m,-1))
    x0 = slices[0] 
    e0 = ll.energy(slices[0], fd)
    grad = ll.gradient(slices[0], fd)[1]
    grad_numeric = np.zeros_like(grad)
    eps =  1e-4j
    for i in range(x0.size):
        x0[i] += eps
        grad_numeric[i] = (ll.energy(x0, fd) - e0)/eps
        x0[i] -= eps

    raise
    # Currently  I assume that the complex part of the gradient is of

    sampler = GibbsOrientationSampler(likelihood=ll,
                                      projection=proj,
                                      quadrature=q,
                                      data=ft_data_sparse)


    x0 = np.fft.fftshift(np.fft.fftn(gt)).ravel() * 1.

    sampler.update_rotations(x0)
    e0 = sampler.energy(x0)
    grad = sampler.gradient(x0)
    grad_num = np.zeros_like(grad)
    eps = 1e-6

    for i in range(x0.size):
        x0[i] += eps
        grad_num[i] = (sampler.energy(x0) - e0)/eps
        x0[i] -= eps


    
