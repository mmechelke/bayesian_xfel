import numpy as np


def project_magnitude(u, constraint, mask= None):
    # Magic number to ensure we do not divide by zero
    eps = 3e-16
    sq_u = np.conj(u) * u
    denom = np.sqrt(sq_u + eps)
    denom2 = sq_u + eps
    r_eps = sq_u/denom - constraint
    dr_eps = (denom2 + eps)/(denom2 * denom)

    if mask is None:
        u_new = (1 - dr_eps * r_eps) *  u
    else:
        u_new = (1 - dr_eps * r_eps) * np.logical_not(mask) * u 
    return u_new

    
def project_support(u, support):
    return support * u


def project_support_and_nonnegativity(u, support):
    return 2 * np.clip(support * u, 0., 1e309) - u


def project_support_and_nonnegativity2(u, support):
    return np.clip(support * u, 0., 1e309)
    
def project_fft_magnitude(u, magnitude, mask=None):
    fu = np.fft.fftn(u)
    pfu = project_magnitude(fu, magnitude,mask)
    return np.fft.ifftn(pfu).real


def update_support_shrink_wrap(magnitude, support, threshold=0.2, blur_std=3.0):
    """
    Updates the support by blurring the current estimate
    of the magnitudes with a gaussian kernel and than updating
    the support based on the threshold
    """
    from scipy.ndimage.filters import gaussian_filter as blur
    blurred_magnitude = blur(magnitude, blur_std)
    magnitude_threshold = blurred_magnitude.max() * threshold

    new_support = blurred_magnitude > magnitude_threshold
    support = np.bitwise_and(new_support, support.astype(bool))

    return support
        
class RAAR(object):

    def __init__(self, beta_0=0.75, beta_max=1., tau = 100):
        self._beta_0 = beta_0
        self._beta_final = beta_max
        self._tau = float(tau)

    def project_support(self, estimate, support):
        u = estimate * support
        return  np.clip(u,0.0,1e309)  

    def project_magnitude(self, estimate, magnitude, mask):
        eps = 3e-16

        mod_sq_u = np.conj(estimate) * estimate + eps
        denom2 = mod_sq_u + eps
        denom = np.sqrt(denom2)
        r_eps = mod_sq_u / denom - magnitude
        dr_eps = (denom2 + eps) / (denom * denom2)
        u_new = (1 - dr_eps * r_eps) * estimate

        return u_new
        
    def run(self, u0, magnitudes, support, niter, mask=None, blur=False, blur_std=1.):
        from scipy.ndimage.filters import median_filter
        from scipy.ndimage.filters import gaussian_filter as blur
        beta = self._beta_0

        u = u0
        fu = np.fft.fftn(u)
        if mask is not None:
            magnitudes[mask==True] = np.abs(fu)[mask==True]
        m_proj = self.project_magnitude(fu, magnitudes, mask=mask)
        u_new = np.fft.ifftn(m_proj).real

        tmp1 = (2 * u_new - u).real

        for i in range(niter):
            beta = np.exp((-i/float(self._tau))**3.0) * self._beta_0 \
                   + (1 - np.exp((-i/self._tau)**3.0)) * self._beta_final

            r_sp = 2 * self.project_support(tmp1, support) - tmp1
            tmp_u = 0.5 * (beta * r_sp + (1 - beta) * tmp1 + u)

            u = tmp_u.real
            fu = np.fft.fftn(u)
            if mask is not None:
                magnitudes[mask==True] = np.abs(fu)[mask==True]
            m_proj =  self.project_magnitude(fu, magnitudes, mask=mask)
            u_new = np.fft.ifftn(m_proj).real

            if blur:
                u_new = blur(u_new, blur_std)
            tmp1 = np.real(2 * u_new - u)

        return self.project_support(u_new, support)

class RAAR_hawk(RAAR):

    def run(self, u0, magnitudes, support, niter, mask=None):
        beta = self._beta_0
        u = u0
        fu = np.fft.fftn(u)
        m_proj = self.project_magnitude(fu, magnitudes)
        u_new = np.fft.ifftn(m_proj).real

        tmp1 = (2 * u_new - u).real

        for i in range(niter):
            beta = np.exp((-i/float(self._tau))**3.0) * self._beta_0 \
                   + (1 - np.exp((-i/self._tau)**3.0)) * self._beta_final

            r_sp = 2 * self.project_support(tmp1, support) - tmp1
            tmp_u = 0.5 * (beta * r_sp + (1 - beta) * tmp1 + u)

            u = tmp_u.real
            fu = np.fft.fftn(u)
            m_proj =  self.project_magnitude(fu, magnitudes, mask=mask)
            u_new = np.fft.ifftn(m_proj).real

            tmp1 = np.real(2 * u_new - u)
        return self.project_support(u_new, support)

class SWRAAR(RAAR):

    def __init__(self, beta_0=0.75, beta_max=1., threshold = 0.1, tau = 100.):
        super(SWRAAR, self).__init__(beta_0, beta_max, tau)
        self._threshold = threshold
        self._blur_std = 1.

    def project_support(self, estimate, support):
        from scipy.ndimage.filters import gaussian_filter as blur
        blurred_magnitude = blur(estimate, self._blur_std)
        magnitude_threshold = blurred_magnitude.max() * self._threshold

        new_support = blurred_magnitude > magnitude_threshold
        support = np.bitwise_and(new_support, support.astype(bool))

        return super(SWRAAR, self).project_support(estimate, support)

        
if __name__ == "__main__":
    import pylab as plt
    import scipy.misc
    import seaborn as sns
    
    lena = scipy.misc.face(gray=True)
    if True:
        plt.imshow(lena, cmap='viridis')
    x_start = np.random.randint(512 - 64)
    y_start = np.random.randint(512 - 64)
    x_start = 540
    y_start = 270
    lena_zeros = np.zeros((128, 128))
    lena_zeros[40:104, 40:104] = lena[x_start:x_start+64, y_start:y_start+64]

    if False:
        plt.figure()
        plt.imshow(lena_zeros)
    
    flena = np.fft.fftn(lena_zeros)
    magnitude = np.sqrt(np.real(flena * np.conj(flena)))

    magnitude = np.fft.ifftshift(np.fft.fftshift(magnitude)
                                 + np.random.normal(size=magnitude.shape) * 1500.)

    mask = np.zeros(magnitude.shape)
    mask[60:80,60:80] = 1.
    magnitude = np.abs(magnitude)
    support = np.zeros((128, 128))
    support[20:108,20:108] = 1.
    magnitude  = magnitude * np.logical_not(mask)
    u0 = np.random.random(lena_zeros.shape)

    raar = RAAR()
    raar._threshold = 0.1
    raar._blur = 2.
    u_reconstruct = raar.run(u0, magnitude, support, 1000, mask=mask)
    u_reconstruct_nomask = raar.run(u0, magnitude, support, 1000,)

    fig = plt.figure()
    ax= fig.add_subplot(131)
    ax.imshow(lena_zeros,cmap='viridis')
    ax= fig.add_subplot(132)
    ax.imshow(u_reconstruct_nomask.real,cmap='viridis')
    ax= fig.add_subplot(133)
    ax.imshow(u_reconstruct.real, cmap='viridis')

    fig = plt.figure()
    ax= fig.add_subplot(121)
    f_m = np.fft.fftn(u_reconstruct)
    f_m = np.sqrt(np.real(f_m * np.conj(f_m)))
    ax.scatter(magnitude.ravel(), f_m.ravel())

    ax= fig.add_subplot(122)
    f_m = np.fft.fftn(u_reconstruct_nomask)
    f_m = np.sqrt(np.real(f_m * np.conj(f_m)))
    ax.scatter(magnitude.ravel(), f_m.ravel())

    
    
