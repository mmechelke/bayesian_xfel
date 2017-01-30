
import numpy as np
import scipy
import re
import os

import hashlib
import csb

from csb.bio.io.wwpdb import StructureParser


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


class ScatteringFactor(object):
    """
    Cacluates the density in reciprocal space as

    F(s) = sum_m f_m(s) exp(-B_m s**2 / 4) exp(i*2pi*s*r)

    where f_m(s) is approximated by four Gaussian distributions
    and exp(-B_m s**2 / 4) are the thermal fluctuations

    g_m(s) = f_m(s) * exp(-B_m s**2 / 4) are precomputed


    """

    def __init__(self, structure=None):
        if structure is None:
            self._atoms = list()
            self._bfactor = list()
            self._seq = list()
            self._elements = list()
        else:
            self._structure = structure
            # For now only non hydrogen atoms
            # TODO use hydrogens as well
            self._atoms = []
            for chain in structure:
                for residue in structure[chain]:
                    for atom in residue:
                        a = residue[atom]
                        if not a.name.startswith("H"):
                            self._atoms.append(residue[atom])
            self._seq = []
            self._bfactor = []
            self._elements = []
            
            for atom in self._atoms:
                self._seq.append(atom.element.name)
                self._elements.append(atom.element.name)

                if atom._bfactor is None:
                    self._bfactor.append(1.)
                else:
                    self._bfactor.append(atom._bfactor)
                    
            self._seq = np.array(self._seq)
            self._elements = set(self._elements)
            self._bfactor = np.clip(self._bfactor, 1., 100.)
            
        self._atom_type_params  = {}
        self._read_sf(fn=os.path.expanduser("~/projects/xfel/py/xfel/core/atomsf.lib"))

    @classmethod
    def from_isd(cls, universe):
        obj = cls()
        atoms = universe.atoms

        for atom in atoms:
            element = str(atom.properties['element'].name)
            obj._elements.append(element)
            obj._atoms.append(atom)
            obj._seq.append(element)
            try:
                obj._bfactor.append(max(1.,atom.properties['bfactor']))
            except KeyError:
                obj._bfactor.append(1.)
        obj._seq = np.array(obj._seq)
        obj._bfactor = np.array(obj._bfactor)
        obj._elements = set(obj._elements)
        obj._bfactor = np.clip(obj._bfactor, 1., 100.)
        return obj
        
    def _read_sf(self, fn):
        """
        Reads the coefficients for the analystical approximation
        to scattering factors from ccp4 database
        """
        float_pattern = '[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        atom_pattern = '[A-Za-z]'
        atom_pattern = '[A-Za-z0-9-+]+'
        line_pattern = ("({0})\s+({1})"
                        "\s+({1})\s+({1})"
                        "\s+({1})\s+({1})"
                        "\s+({1})\s+({1})"
                        "\s+({1})\s+({1})").format(atom_pattern,float_pattern)

        regex = re.compile(line_pattern)


        with open(fn) as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    continue

                m = regex.match(line)
                atom_name = m.groups()[0]
                a1, a2, a3, a4 = m.groups()[1], m.groups()[3], m.groups()[5], m.groups()[7]
                b1, b2, b3, b4 = m.groups()[2], m.groups()[4], m.groups()[6], m.groups()[8]
                c = m.groups()[9]

                a = np.array([a1,a2,a3,a4],np.double)
                b = np.array([b1,b2,b3,b4],np.double)

                self._atom_type_params[atom_name] = (a,b,float(c))

    def _calculate_gm(self, hkl):
        """
        calculates the the product of scattering factor and
        debye-waller factors
        """
        f = np.zeros((len(self._atoms), hkl.shape[0]))
        seq = self._seq
        bfactor = self._bfactor

        s_tols = 0.25 * (hkl**2).sum(-1)

        for atom_type in self._elements:
            a,b,c = self._atom_type_params[atom_type]
            indices = np.where(seq==atom_type)[0]
            fx = c + np.dot(np.exp(np.outer(-s_tols,b)),a)
            f[indices,:] = fx[:]

        f *= np.exp(np.outer(-bfactor,s_tols))

        return f

    def _calculate_gm_grad(self, hkl):
        """
        calculate the gradien of the scattering factor and
        debye-waller factor
        """
        seq = np.array([a.element.name for a in self._atoms])
        f = np.zeros((len(self._atoms), hkl.shape[0]))
        dfg = np.zeros((len(self._atoms), hkl.shape[0], 3))

        bfactors = np.array([a.bfactor for a in self._atoms])
        bfactors = np.clip(bfactors, 1., 100.)
        s_tols = 0.25 * (hkl**2).sum(-1)

        for atom_type in self._elements:
            a,b,c = self._atom_type_params[atom_type]
            indices = np.where(seq==atom_type)[0]
            bfactor = bfactors[indices]
            g = np.exp(np.outer(-s_tols,b))
            sf = np.dot(g, a) + c
            gsf = np.sum(g * a[np.newaxis,:] * b[np.newaxis,:] * -0.5, -1)

            dwf = np.exp(-np.outer(bfactor, s_tols))
            gdwf = dwf * (bfactor * - 0.5)[:,np.newaxis]

            grad = sf * gdwf + gsf * dwf

            f[indices,:] = dwf * sf
            dfg[indices,:,:] = grad[:,:,np.newaxis] * hkl

        return dfg, f


    def _calculate_scattering_factors(self, hkl):
        """
        creates an approximation of the density in reciprocal space by
        four gaussians

        returns the scattering vectors
        """
        seq = self._seq
        bfactor = self._bfactor
        f = np.zeros((len(self._atoms), hkl.shape[0]))

        s_tols = 0.25 * (hkl**2).sum(-1)

        for atom_type in self._elements:
            a,b,c = self._atom_type_params[atom_type]
            indices = np.where(seq==atom_type)[0]
            fx = c + np.dot(np.exp(np.outer(-s_tols,b)),a)
            f[indices,:] = fx[:]

        return f

    def _calculate_debyewaller_factors(self, hkl):
        """
        """
        b  = np.array(self._bfactor)
        s_tols = 0.25 * (hkl**2).sum(-1)
        t = np.exp(np.outer(-b,s_tols))

        return t

    def grad_s(self, X, hkl):
        """
        Gradient with respect to the reciprocal space coordinates

        @param X: atomic positions
        @param hkl: reciprocal space positions

        """
        seq = np.array([atom.element.name for atom in self._atoms])
        bfactor = np.array([atom.bfactor for atom in self._atoms])
        bfactor = np.clip(bfactor, 1., 100.)

        s_tols = 0.25 * (hkl**2).sum(-1)
        dw_factors = np.exp(np.outer(-bfactor, s_tols))


    def grad_hkl(self, X, hkl):
        seq = self._seq
        bfactor = self._bfactor
        bfactor = np.clip(bfactor, 1., 100.)

        dg = np.zeros((len(self._atoms), hkl.shape[0], hkl.shape[1]))
        g = np.zeros((len(self._atoms), hkl.shape[0]))
        s_tols = 0.25 * (hkl**2).sum(-1)

        dw_factors = np.exp(np.outer(-bfactor, s_tols))
        ddw_factors = bfactor[:,np.newaxis] * dw_factors

        for atom_type in self._elements:
            a,b,c = self._atom_type_params[atom_type]
            indices = np.where(seq==atom_type)[0]
            inner_exp = np.exp(np.outer(-s_tols,b))
            sf = np.dot(inner_exp, a) + c
            dsf = np.dot(inner_exp, a*b)
            gx = dsf * dw_factors[indices] + sf * ddw_factors[indices]
            g[indices,:] = sf[:] * dw_factors[indices]
            a = np.einsum('ab,bc->abc',gx, -0.5*hkl)
            dg[indices,:,:] = a

        phase = np.dot((2 * np.pi * X),hkl.T)
        fx=  np.sum(g * np.exp(1j * phase),0)

        g2 = np.einsum('ba,bc->bac',g , 2 * np.pi * 1j *X)
        dfx = np.einsum("abc,ab->bc",dg + g2,np.exp(1j * phase))

        return dfx, fx

    
        
    def calculate_structure_factors(self, X, hkl):
        """
        TODO do this calculation in chunks to save space
        """
        F = np.zeros(hkl.shape[0], dtype=np.complex128)

        lim = hkl.shape[0]
        step = 512
        for i in range(0,lim,step):
            _hkl = hkl[i:i+step]
            f = self._calculate_scattering_factors(_hkl)
            f *= self._calculate_debyewaller_factors(_hkl)
            phase = np.dot((2 * np.pi * X),_hkl.T)
            F[i:i+step] = np.sum(f * np.exp(1j * phase),0)
        return F

    def calculate_structure_factor_gradient(self, X, hkl):
        """
        calculates the gradient of the fourier density 
        with respect to the atomic coordinates
        """
        G = np.zeros(hkl.shape, dtype=np.complex128)
        lim = hkl.shape[0]
        F = np.zeros(hkl.shape[0], dtype=np.complex128)

        step = 512

        for i in range(0, lim, step):
            _hkl = hkl[i:i+step]

            dfg, f = self._calculate_gm_grad(_hkl)

            phase = np.exp(1j * np.dot((2 * np.pi * X), _hkl.T))
            gphase = phase[:, :, np.newaxis] *\
                     1j * 2 * np.pi * X[:, np.newaxis, :]
            grad = dfg * phase[:, :, np.newaxis] 
            grad += f[:, :, np.newaxis] * gphase
            F[i: i+step] = np.sum(f * phase, 0)
            G[i: i+step, :] = np.sum(grad, 0)

        return G, F


    def calculate_structure_factor_gradient2(self, X):
        """
        calculates the gradient of the fourier density 
        with respect to the atomic coordinates 
        """
        g_m = self._calculate_scattering_factors(hkl)
        g_m *= self._calculate_debyewaller_factors(hkl)

        phase = np.dot((2 * np.pi * X),self._hkl.T)
        fx = (g_m *1j * 2 * np.pi  * np.exp(1j * phase))
        dF_dx =  np.array([np.multiply.outer(s,fx_s) for s,fx_s in
                           zip(fx.T,self._hkl)])

        return dF_dx

    def calculate_intensity_gradient(self, X):
        """
        calculates the gradient of the intensity with respect to the atomic coordinates dI/dx
        """
        g_m = self._calculate_scattering_factors(self._hkl)
        g_m *= self._calculate_debyewaller_factors(self._hkl)
        phase = np.dot((2 * np.pi * X),self._hkl.T)
        F = np.sum(g_m * np.exp(1j * phase),0)
        fx = (g_m *1j * 2 * np.pi  * np.exp(1j * phase))
        dF_dx =  np.array([np.multiply.outer(s,fx_s) for s,fx_s in zip(fx.T,self._hkl)])
        dI_dx =  np.conj(F[:,np.newaxis,np.newaxis]) * dF_dx + F[:,np.newaxis,np.newaxis] * np.conj(dF_dx)

        return dI_dx


class Correlations(object):

    def __init__(self, angles, nbins):

        self._bin_angles(angles, nbins)

    def _bin_angles(self, angles, nbins):
        pass

    def calculate_from_density(self, rho):
        pass


class OnePhotonCorrelations(Correlations):

    def _bin_angles(self, angles, nbins):
        d = np.sqrt(np.sum(angles**2,-1))
        lower = d.min()
        upper = d.max()

        axes = np.linspace(lower, upper, nbins)
        indices = np.argsort(d)
        bins = [[] for x in xrange(nbins)]

        j = 0
        for i in range(0,axes.shape[0]):
            right_edge = axes[i]
            print right_edge, i
            while d[indices[j]] < right_edge:
                bins[i-1].append(indices[j])
                j += 1

        bins[-1] = indices[j:].tolist()

        self._axes  = axes
        self._bins = bins

    def calculate_from_density(self, rho):
        I = np.asarray([np.sum(rho.take(bin))
                        for bin in self._bins])

        return I




class CachedScatteringFactor(ScatteringFactor):

    def __init__(self, structure):
        super(CachedScatteringFactor,self).__init__(structure)

        self._f = None

    def calculate_structure_factors(self, X, hkl):
        if self._f is None:
            print "calc f"
            self._f = self._calculate_scattering_factors(hkl)
            self._f *= self._calculate_debyewaller_factors(hkl)
        else:
            print "using cached f"

        phase = np.dot((-2 * np.pi * X),hkl.T)
        F = np.sum(self._f * np.exp(1j * phase),0)

        return F



class SphericalSection(object):

    def get(self,
            n_points=20, radius=1.0,
            polar_min=0., polar_max=np.pi,
            azimut_min=0., azimut_max=2*np.pi):

        theta = np.linspace(polar_min,polar_max, n_points)
        phi = np.linspace(azimut_min, azimut_max, n_points)

        x = np.outer(radius*np.sin(theta), np.cos(phi))
        y = np.outer(radius*np.sin(theta), np.sin(phi))
        z = np.outer(radius*np.cos(theta), np.ones(n_points))

        return [x,y,z]


class EwaldSphereProjection(object):


    def get_indices(self, wavelength, x,y,z):
        """
        projects dectector points onto an Ewald Sphere
        x, y, z are the pixel coordinates

        x, y, z are all M x N matrices, where M x N is the detector size.

        It is assumed that the detector is perpendicular to the Z-axis

        """

        d = np.sqrt(x**2 + y**2 + z**2)

        h = 1/wavelength * (x/d)
        k = 1/wavelength * (y/d)
        l = 1/wavelength * (z/d)

        return h,k,l


    def project(self, structure_factor, angle):
        pass









if __name__ == "__main__":
    import matplotlib
    matplotlib.interactive(True)

    import time
    import os
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import pylab
    from pylab import *

    from csb.bio.io.wwpdb import StructureParser
    from csb.bio.io.wwpdb import get
    from xfel.core.density import Density

    #structure = get("1L2Y")
    #structure = StructureParser(os.path.expanduser("~/data/pdb/caffeine2.pdb")).parse()

    #fn = os.path.expanduser("~/gsh.pdb")
    structure = StructureParser(os.path.expanduser("~/projects/xfel/data/GTT_short.pdb")).parse()

    x = np.linspace(-1.,1.,11)
    h, k, l = np.meshgrid(x,x,x)
    hkl = np.vstack([item.ravel() for item in [h,k,l]]).T
    hkl = np.ascontiguousarray(hkl)
    bf = np.random.random()

    def bfactors(hkl, bf):
        return np.exp(-0.25 * bf * (hkl**2).sum(-1))

    def bfactor_grad(hkl):
        return np.exp(-0.25 * bf * (hkl**2).sum(-1))[:,np.newaxis] * -0.5 * hkl * bf

    a = np.random.random(4,)
    b = np.random.random(4,)
    c = 0.3

    def sf(hkl,a,b,c):
        s_tols = -0.25 * (hkl**2).sum(-1)
        inner_exp = np.exp(np.outer(-s_tols,b))
        sf = np.dot(inner_exp, a) + c

        return sf

    def sf_grad(hkl, a, b, c):
        s_tols = -0.25 * (hkl**2).sum(-1)
        sf = np.exp(np.outer(-s_tols,b)) * a[np.newaxis,:] * b[np.newaxis,:] * 0.5

        return sf.sum(-1)[:,np.newaxis] * hkl

    def gm(hkl, a, b, c, bf):
        s_tols = -0.25 * (hkl**2).sum(-1)
        inner_exp = np.exp(np.outer(-s_tols,b))
        sf = np.dot(inner_exp, a) + c
        bf = np.exp(bf * s_tols)

        return sf * bf

    def gm_grad(hkl, a, b, c, bf):
        s_tols = -0.25 * (hkl**2).sum(-1)
        g = np.exp(np.outer(-s_tols,b))
        sf = np.dot(g, a) + c
        gsf = np.sum(g * a[np.newaxis,:] * b[np.newaxis,:] * 0.5, -1)

        bb = np.exp(bf * s_tols)
        gb = bb * bf * - 0.5
        grad = sf * gb  + gsf * bb
        return grad[:,np.newaxis] * hkl

    sf = ScatteringFactor(structure)
    X = np.array([a.vector for a in  sf._atoms])
    X -= X.mean(0)
    if False:

        n = 10
        X = X[:n]
        sf._seq = sf._seq[:n]
        sf._elements = ['N', 'C']
        sf._atoms = sf._atoms[:n]
        sf._bfactor = sf._bfactor[:n]

    dgm, f1 = sf._calculate_gm_grad(hkl)

    f = sf._calculate_scattering_factors(hkl)
    f *= sf._calculate_debyewaller_factors(hkl)

    scatter(f.real.ravel(), f1.real.ravel())

    dgm2 = dgm * 0.0
    eps = 1e-7
    for i in range(3):
        hkl[:, i] += eps
        fprime = sf._calculate_scattering_factors(hkl)
        fprime *= sf._calculate_debyewaller_factors(hkl)
        dgm2[:, :, i] = (fprime - f)/eps
        hkl[:, i] -= eps

    figure()
    scatter(dgm.real.ravel(), dgm2.real.ravel())
    
    G, FF = sf.calculate_structure_factor_gradient(X, hkl)
    G2 = G * 0.0
    F = sf.calculate_structure_factors(X, hkl)
    eps = 1e-7
    for i in range(3):
        hkl[:,i] += eps
        
        G2[:,i] = (sf.calculate_structure_factors(X, hkl) - F)/eps
        hkl[:,i] -= eps

    figure()
    scatter(G.real.ravel(), G2.real.ravel())
    scatter(G.imag.ravel(), G2.imag.ravel())

    figure()
    scatter(F.real.ravel(), FF.real.ravel())
    show()

    t0 = time.time()
    G, FF = sf.calculate_structure_factor_gradient(X, hkl)
    print "hkl gradient: {} \n".format(time.time() - t0)
    t0 = time.time()
    g = sf.grad_hkl(X, hkl)
    print "X gradient: {} \n".format(time.time() - t0)

    raise
    sf = ScatteringFactor(structure)
    sf._hkl = hkl

    X = np.array([a.vector for a in  sf._atoms])
    X -= X.mean(0)

    g,g2 = sf.grad_hkl(X, hkl)
    F = sf.calculate_structure_factors(X,hkl)
    gi= sf.calculate_intensity_gradient(X)

    raise

    F = F.reshape(h.shape)
    rho = np.fft.fftshift(np.abs(np.fft.ifftn(F,[250,250,250])))

    grid = Density.from_voxels(np.abs(F)**2,1.)
    grid.write_gaussian(os.path.expanduser("~/mr.cube"))
    raise
    grid = Density.from_voxels(rho,1.)
    grid.write_gaussian(os.path.expanduser("~/mr2.cube"))


    raise



    if True:
        fig = pylab.figure()
        ax = fig.add_subplot(131)
        xi, yi= np.mgrid[0:500:1,0:500:1]
        ax.contour(xi,yi, rho.sum(0), 30)
        pylab.show()
        ax = fig.add_subplot(132)
        xi, yi= np.mgrid[0:500:1,0:500:1]
        ax.contour(xi,yi, rho.sum(1), 30)
        pylab.show()
        ax = fig.add_subplot(133)
        xi, yi= np.mgrid[0:500:1,0:500:1]
        ax.contour(xi,yi, rho.sum(2), 30)
        pylab.show()

    raise

    from mayavi import mlab
    xi, yi, zi = np.mgrid[0:500:1,0:500:1,0:500:1]
    obj = mlab.contour3d(rho, contours=10, transparent=True)
    mlab.show()

    from mayavi import mlab
    obj = mlab.contour3d(np.abs(F), contours=10, transparent=True)
    mlab.show()

    raise
    for ii in range(0,F.shape[0],25):
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        xi, yi= np.mgrid[0:500:1,0:500:1]
        ax.contour(xi,yi,rho[ii,:,:], 30)
        pylab.show()

    I = np.abs(F)**2

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    nx, ny, nz = I.shape
    xi, yi= np.mgrid[0:nx:1,0:ny:1]
    ax.contour(xi,yi, I.sum(2), 15)
