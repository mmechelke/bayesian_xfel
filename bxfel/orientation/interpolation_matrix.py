import numpy as np
import scipy.sparse as sp

from joblib import Memory
import os
memory = Memory(cachedir=os.path.expanduser("~/tmp/joblib"),
                verbose=0)

def _get_coords(n_voxel, radial_cutoff=None, ndim=2):
    if ndim==2:
        yv, xv = np.meshgrid(np.arange(n_voxel)-(n_voxel)/2,
                             np.arange(n_voxel)-(n_voxel)/2)

        coords = np.array([[xx, yy, 0.]
                           for xx, yy in zip(xv.ravel(), yv.ravel())])

    elif ndim==3:
        yv, xv, vv = np.meshgrid(np.arange(n_voxel)-(n_voxel)/2,
                                 np.arange(n_voxel)-(n_voxel)/2,
                                 np.arange(n_voxel)-(n_voxel)/2)
        coords = np.array([[xx, yy, vvv]
                           for xx, yy, vvv in zip(xv.ravel(),
                                                yv.ravel(),
                                                vv.ravel())])
    else:
        raise NotImplementedError("Not supported")

    if radial_cutoff is not None:
        sq_radius = np.sum(coords**2, -1)
        in_range = sq_radius < (radial_cutoff * n_voxel/2.)**2
        coords = coords[in_range]

    return coords

def place_pixel_on_ewald_sphere(x, y, zL):
    v =  np.array([x,y,zL]) - np.sqrt(1 + (x*x + y*y)/(zL*zL)) - np.array([0,0,zL])
    return v


def get_coords_on_eqald_sphere(n_voxel, radial_cutoff=None,
                               detector_distance=1.,
                               pixel_size=1.):
    zL = detector_distance / pixel_size
    zL_sq = zL*zL
    yv, xv = np.meshgrid(np.arange(n_voxel) - (n_voxel)/2 + 1,
                         np.arange(n_voxel) - (n_voxel)/2 + 1)

    coords = np.array([np.array([xx,yy,zL]) - np.sqrt(1 + (xx*xx + yy*yy) / (zL_sq))
                       for xx, yy in zip(xv.ravel(), yv.ravel())])
    coords -=  np.array([0,0,zL])

    return coords 


def get_plane_coords(n_voxel, radial_cutoff=None):
    return _get_coords(n_voxel, radial_cutoff, ndim=2)

def get_cube_coords(n_voxel, radial_cutoff=None):
    return _get_coords(n_voxel, radial_cutoff, ndim=3)


def get_sparse_to_complete_projection(n_voxel, radial_cutoff, ndim=2):
    """
    Generates a sparse matrix that maps a sparse image vector to a full
    N X N vector

    @param n_voxel: number of pixel in each dimension
    @param radial_cutoff: radial cutoff 1<0

    """

    coords = _get_coords(n_voxel, ndim=ndim)

    sq_radius = np.sum(coords**2, -1)

    if radial_cutoff is not None:
        in_range = sq_radius < (radial_cutoff * n_voxel/2.)**2
    else:
        in_range = np.ones(sq_radius.shape).astype(np.bool)
    n_active_pixel = np.sum(in_range)

    splil = sp.lil_matrix((n_voxel**int(ndim), n_active_pixel),
                          dtype=np.float32)

    j = 0
    for i, value in enumerate(in_range):
        if value:
            splil[i, j] = 1.0
            j += 1

    sparse_mat = splil.tocsr()
    sparse_mat.eliminate_zeros()
    return sparse_mat

def get_sparse_to_image_projection(n_voxel, radial_cutoff=1.):
    return get_sparse_to_complete_projection(n_voxel, radial_cutoff, ndim=2)

def get_image_to_sparse_projection(n_voxel, radial_cutoff=1.):
    return get_sparse_to_complete_projection(n_voxel, radial_cutoff, ndim=2).T

def get_sparse_to_cube_projection(n_voxel, radial_cutoff=1.):
    return get_sparse_to_complete_projection(n_voxel, radial_cutoff, ndim=3)

def get_cube_to_sparse_projection(n_voxel, radial_cutoff=1.):
    return get_sparse_to_complete_projection(n_voxel, radial_cutoff, ndim=3).T


def generate_coords(num, dim, start=None, stop=None, radial_cutoff=None,):
    """
    generates coordiantes of evenly spaced points over the spcified intervals

    @param num: Number of points in each dimension
    @param dim: Dimension of the points
    @param start: The starting value in each dimension
    @param stop: The last value in each dimension.
    @param radial_cutoff: fraction of the max radius to truncate the coordinate array

    """
    if start is None:
        start = -float(num)/2

    if stop is None:
        stop = float(num)/2

    x = np.linspace(start, stop, num)
    params = (x,) * dim
    g = np.meshgrid(*params)
    positions = np.vstack(map(np.ravel, g))

    if radial_cutoff is not None:
        sq_radius = np.sum(positions**2, -1)
        mask = sq_radius < (radial_cutoff * start)**2
        positions = positions[mask,:]

    return positions.T


def compute_trilinear_weights(x, lattice):
    """
    Assumes that the lattice points in 3D  form a regular hexadron
    with side length one and its origin is at (0,0,0) and that x is
    within the lattice

    @param x: interpolation point
    @param lattice: lattice points
    """
    weights = np.zeros(8)

    for i, lattice_vertex in enumerate(lattice):
        a, b, c = lattice_vertex
        weights[i] = ((a * x[0] + (1-a) * (1-x[0])) *
                      (b * x[1] + (1-b) * (1-x[1])) *
                      (c * x[2] + (1-c) * (1-x[2])))
    return weights


def get_indices(hex_index, dims):
    try:
        return np.ravel_multi_index(hex_index, dims)
    except ValueError:
        pass

@memory.cache
def compute_interpolation_matrix(coords, n_voxel, n_pixel = None):
    """
    We assume that all coordinates are contained within the grid
    and that the grid is shaped n_voxel x n_voxel x n_voxel with
    the orgin at the center and a spacing of one

    we assume that the sizes are equal, thus only the voxel size
    is changing

    """

    if n_pixel is None:
        n_pixel = n_voxel
        spacing = 1.
    else:
        spacing = float(n_pixel)/n_voxel
    hexadron = generate_coords(2, 3, 0., 1.)
    hexadron = hexadron.astype(np.int)

    row = []
    col = []
    values = []

    for i, point in enumerate(coords):

        ref_p = np.floor(point/spacing)
        hex_index = (hexadron + ref_p + n_voxel/2) 
        hex_index = hex_index.astype(np.int)

        in_range = np.all(hex_index < n_voxel, 1)
        ind = np.ravel_multi_index(hex_index.T,
                                   (n_voxel, n_voxel, n_voxel),
                                   mode='clip')

        ind = ind[in_range]
        weights = compute_trilinear_weights(point/spacing - ref_p, hexadron)
        weights = weights[in_range]

        values.extend(weights)
        row.extend(np.ones(len(weights), dtype=np.int) * i)
        col.extend(ind)

    interpolation_matrix = sp.coo_matrix((values, (row, col)), shape=(len(coords), n_voxel**3))
    interpolation_matrix = interpolation_matrix.tocsr()
    interpolation_matrix.eliminate_zeros()

    return interpolation_matrix

@memory.cache
def compute_slice_interpolation_matrix(rotations, n_voxel, n_pixel=None,
                                       radius_cutoff=None,
                                       symmetries=None):
    """
    Computes a sparse projection of the n x n x n volume on m slices
    with radius radius_cutoff and symmetry relations defines by the
    projections matrices defines in symmetries
    """

    matrices = []
    if n_pixel is None:
        n_pixel = n_voxel

    coords = get_plane_coords(n_pixel, radius_cutoff)

    for rotation in rotations:
        if symmetries is None:
            matrices.append(compute_interpolation_matrix(np.dot(coords, rotation),
                                                         n_voxel, n_pixel))
        else:
            tmp = []
            for sym in symmetries:
                tmp.append(compute_interpolation_matrix(np.dot(coords, np.dot(sym, rotation)),
                                                             n_voxel, n_pixel))
            m = np.sum(tmp, 0)
            m /= len(symmetries)

            matrices.append(m)

    return sp.vstack(matrices)


if __name__ == "__main__":

    from bxfel.io import mrc
    import pylab as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D
    import os


    n_voxel = 101
    c = get_coords_on_eqald_sphere(n_voxel, radial_cutoff=None,
                                   detector_distance=.13,
                                   pixel_size=0.0014)
    fig  = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(c[:,0],c[:,1],c[:,2])

    ax.set_zlim((-50,50))

    cx = compute_interpolation_matrix(c, 101,101)
    a = np.ones(cx.shape[0])
    vol  = cx.T.dot(a)
    vol = vol.reshape((n_voxel, n_voxel, n_voxel))
    mrc.write(vol, os.path.expanduser("~/x.mrc"))

    raise
    c = generate_coords(2,3,0.,1.)
    c = np.random.random(size = (2000,3)) * 16  - 8

    x = np.arange(n_voxel**3).reshape((n_voxel,n_voxel,n_voxel))
    x = x.astype(np.double)

    xx = np.ones(n_voxel**3, dtype=np.float32)
    proj = get_cube_to_sparse_projection(n_voxel, 0.9)
    yy = proj.T.dot(proj.dot(xx)).reshape((n_voxel, n_voxel, n_voxel))

    mrc.write(yy, os.path.expanduser("~/test_sparse.mrc"))

    w = compute_interpolation_matrix(c, n_voxel)

    a = w.dot(x.ravel())

    from scipy import ndimage
    b =  ndimage.map_coordinates(x, c.swapaxes(0,1) + 8, order=1)

    resolution = 17
    n_data = 3000
    rad = 0.95

    ground_truth = mrc.read(os.path.expanduser("~/projects/xfel/data/phantom/phantom.mrc"))[0]
    ground_truth = ndimage.zoom(ground_truth, resolution/128.)

    q = ChebyshevSO3Quadrature(4)
    m = len(q.R)

    cc = get_plane_coords(resolution)

    w = compute_interpolation_matrix(cc, 17)

    p = w.dot(ground_truth.ravel())
    p = p.reshape((resolution,resolution))

    gt = ground_truth[:,:,n_voxel/2]

    plt.figure()
    plt.imshow(p)
    plt.figure()
    plt.imshow(gt)

    plt.figure()
    plt.plot(gt.ravel())
    plt.plot(p.ravel())

    c1 = get_plane_coords(121,None)[:,:2]

    c2 = get_plane_coords(121,1.)[:,:2]

    plt.figure()
    plt.scatter(c1[:,0],c1[:,1])
    plt.scatter(c2[:,0],c2[:,1], c='r')


    cutoff = 0.9
    symmetries = None

    proj = compute_slice_interpolation_matrix(q.R, n_voxel, cutoff, symmetries)
    to_imag = get_sparse_to_image_projection(n_voxel, cutoff)
    g = proj.dot(ground_truth.ravel())

    g = g.reshape((len(q.R), -1))
    data = np.array(map(to_imag.dot, g))

    from xfel.grid import Grid

    mygrid = Grid(np.arange(-8,9), np.arange(-8,9), np.arange(-8,9), ground_truth)

    fig  = plt.figure(figsize=(16,9))

    for i in range(24):
        ax = fig.add_subplot(4,6,i+1)
        ax.imshow(data[i].reshape((n_voxel, n_voxel)),
                  cmap='viridis')

    fig  = plt.figure(figsize=(16,9))
    for i in range(24):
        ax = fig.add_subplot(4,6,i+1)
        s = mygrid.slice(q.R[i])
        ax.imshow(s.T,
                  cmap='viridis')


    fig  = plt.figure(figsize=(16,9))
    for i in range(24):
        ax = fig.add_subplot(4,6,i+1)
        s = mygrid.slice(q.R[i]).T

        ax.plot(data[i].reshape((n_voxel, n_voxel)).ravel())
        ax.plot(s.ravel())






