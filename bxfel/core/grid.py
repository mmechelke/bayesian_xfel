##
## 3D cubic grid
##
## Author: Michael Habeck, Martin Mechelke
import numpy as np

"""
3D cubic orthogonal grid supporting convolution with Gaussian kernel
"""

class Grid(object):
    '''Class for holding density map data.'''


    def __init__(self, nx, ny, nz, values=None, spacing=1.):
        '''Creates a grid instance.

        @param nx: number of grid points in x direction
        @type nx: int
        @param ny: number of grid points in y direction
        @type ny: int
        @param nz: number of grid points in z direction
        @type nz: int

        @param spacing: the spacing between two grid points
        @type spacing: float

        '''
        from bxfel._bxfel import grid

        self.ctype = grid((int(nx),int(ny),int(nz)))
        self.spacing = float(spacing)
        if values is not None:

            assert len(values.shape) == 3

            self.set_rho(values.astype(np.double))

    def __getattr__(self, attr):

        if hasattr(self.ctype, attr):
            return getattr(self.ctype, attr)

        elif attr == 'shape':
            return self.nx, self.ny, self.nz
        return object.__getattribute__(self, attr)


    def __setattr__(self, attr, value):
        if attr == 'origin':
            if len(value) != 3:
                raise ValueError('<origin> must have three components.')
            self.ctype.origin = tuple(value)
        elif attr == 'spacing':
            self.ctype.spacing = value
        elif attr == 'n_shells':
            self.ctype.n_shells = value
        else:
            object.__setattr__(self, attr, value)

    def __str__(self):
        return "Grid(nx=%d, ny=%d, nz=%d, spacing=%.2f)" \
               % (self.nx, self.ny, self.nz, self.spacing)

    __repr__ = __str__

    def __getstate__(self):

        return self.origin, self.shape, self.spacing, self.values

    def __iadd__(self, other):

        if self.shape != other.shape:
            raise ValueError('shape mismatch; unable to add these grids.')
        self.set_rho(self.values + other.values)
        return self

    def __isub__(self, other):

        if self.shape != other.shape:
            raise ValueError('shape mismatch; unable to subtract these grids.')
        self.set_rho(self.values - other.values)
        return self

    def __add__(self, other):

        if self.shape != other.shape:
            raise ValueError('shape mismatch; unable to add these grids.')
        new_grid = Grid(*self.shape)
        new_grid.spacing = self.spacing
        new_grid.origin = self.origin
        new_grid.set_rho(self.values + other.values)

        return new_grid

    def __sub__(self, other):

        if self.shape != other.shape:
            raise ValueError('shape mismatch; unable to subtract these grids.')
        new_grid = Grid(*self.shape)
        new_grid.spacing = self.spacing
        new_grid.origin = self.origin
        new_grid.set_rho(self.values - other.values)

        return new_grid

    def __deepcopy__(self, memo={}):

        state = self.__getstate__()
        grid  = Grid(*state[1])
        grid.__setstate__(state)

        return grid

    __copy__ = __deepcopy__
    
    def __setstate__(self, state):

        Grid.__init__(self, *state[1])
        self.spacing = float(state[2])
        self.origin  = tuple(map(float,state[0]))
        self.set_rho(state[3])

    def clone(self):
        '''Creates a clone of self.

        @return: a clone of self.
        '''
        state = self.__getstate__()
        copy = Grid(*state[1])
        copy.spacing = float(state[2])
        copy.origin  = tuple(map(float,state[0]))
        copy.set_rho(np.reshape(state[3], state[1]))

        return copy
        
    def get_data(self):
        '''Returns the density data of the grid.

        In contrast to using grid.values directly this reshapes the data to the
        grids\' shape.

        @return: a (nx, ny, nz)-shaped array containing the density data
        '''
        from numpy import reshape
        return reshape(self.values, self.shape)

    def set_data(self, values):
        """
        Sets the values of the grid to values

        takes care of reshaping 
        """
        self.set_rho(np.reshape(values, (self.shape)))


    def add(self, other):
        '''Adds the density of other to the one of self.
        Self will contain the sum of densities afterwards.

        In contrast to using \'self += other\', this does not simply
        add the data arrays (i.e. disregarding the spatial position of the
        grid), but uses the density that really lies in the same region of space
        by interpolation.

        @param other: the grid whose density will be subtracted from self.
        @type other: a Grid instance
        '''
        from numpy import zeros, identity
        new_grid = self.clone()
        other.transform_and_interpolate(identity(3), zeros(3), new_grid.ctype)
        self += new_grid

    def axis(self, axis=0):
        from numpy import arange
        return self.origin[axis] + arange(self.shape[axis]) * self.spacing


    def subtract(self, other, allow_negative=False):
        '''Subtracts the density of other from the one of self.
        Self will contain the difference afterwards.

        In contrast to using \'self -= other\', this does not simply
        subtract the data arrays (i.e. disregarding the spatial position of the
        grid), but uses the density that really lies in the same region of space
        by interpolation.

        @param other: the grid whose density will be subtracted from self.
        @type other: a Grid instance

        @param allow_negative: If False, every value below 0 will be set to 0.
        @type allow_negative: bool
        '''
        from numpy import zeros, identity

        new_grid = self.clone()
        new_grid.set_rho(0.*new_grid.values)
        other.transform_and_interpolate(identity(3), zeros(3), new_grid.ctype)

        new_values = self.values - new_grid.values

        if not allow_negative:
            import sys
            '''check Python version, as sys.float_info if only available in 2.6
            and newer.'''
            major_version, minor_version = map(int, sys.version.split('.')[:2])

            max_float = float(sys.maxint) ## for Python version < 2.6
            if major_version > 2 or \
                   (major_version == 2 and minor_version >= 6):
                max_float = sys.float_info.max ## for version >= 2.6

            new_values = new_values.clip(0, max_float)
            
        self.set_rho(new_values)

    def cc(self, R, t, other):
        '''Returns the correlation between self and another grid intance.
        A rotation and translation must be given as arguments and will be applied
        to the other grid before the correlation computation.

        @param R: the rotation applied to the other grid before correlation
        computation.
        @type R: a 3x3 array

        @param t: the translation applied to the other grid before correlation
        computation.
        @type t: a 3 element array
        '''        
        from numpy import array
        return self.ctype.cc(array(R)*1.,array(t)*1.,other.ctype)
        
    def set_n_shells(self, precision = 7):

        from numpy import sqrt, ceil
        
        log10 = 2.3025850929940459
        n_shells = ceil(sqrt(precision*log10)/ self.spacing)
        self.n_shells = int(n_shells)

        
    def center_of_mass(self):
        '''Returns the center of mass of the grid.

        @return: the center of mass of the grid.
        '''
        from numpy import dot, array, sum

        x = [dot(self.project1D(i), self.axis(i)) for i in range(3)]
        return array(x) / sum(self.values)

    def median_of_mass(self):
        '''Returns the median of mass of the grid.

        @return: the median of mass of the grid.
        '''
        from numpy import weighted_median, array

        x = [weighted_median(self.axis(i), self.project1D(i))
             for i in range(3)]
        return array(x)

    def profile(self, origin = None, binsize = 0.1):

        from numpy import sqrt, transpose, arange

        if origin is None:
            origin = self.center_of_mass()

        x,y,z = [(self.axis(i)-origin[i])**2 for i in range(3)]

        rho  = self.values
        hist = {}

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):

                    d = sqrt(x[i]+y[j]+z[k])
                    n = int(d/binsize)

                    index = k + self.nz * (j + self.ny * i)
                    hist[n] = hist.get(n, 0.) + rho[index]

        x = arange(max(hist.keys()))
        y = [hist.get(i,0.) for i in x]

        return transpose([x*binsize, y])


    def center(self):
        '''Shifts the grid so that its origin is in the center of mass.'''
        from numpy import array
        self.origin = tuple(array(self.origin)-self.center_of_mass())


    def rotate(self, R):
        '''Returns a rotated copy of the grid.
 
        @param R: the rotation matrix.
        @type R: a 3x3 array

        @return: a copy of the grid rotated by <R>.
        '''
        from numpy import array, dot, ceil, maximum, minimum, transpose

        ## find new bounding box

        o = array(self.origin)
        n = array(self.shape) * self.spacing
        
        corners = [o + n * array([i,j,k]) for i in range(2)
                   for j in range(2) for k in range(2)]
        
        corners = dot(array(corners),transpose(R))

        lower = minimum.reduce(corners,0)
        upper = maximum.reduce(corners,0)
        
        n = ceil((upper-lower)/self.spacing).astype('i')

        grid = Grid(n[0], n[1], n[2], spacing=self.spacing)
        grid.origin = tuple(lower.tolist())

        self.transform_and_interpolate(R, 0. * lower, grid.ctype)

        return grid

    def translate(self, t, create_new_instance=True):
        '''Creates a translated version of self.
        If create_new_instance is True, self is cloned and the copy is
        translated and returned. Else, the origin of self will be translated and
        nothing is returned.

        @param t: a 3-element translation vector (in Angstroms).
        @type t: iterable

        @param create_new_instance: If True copies self and applies the
        translation to the copy. Else the origin of self is translated.
        @type create_new_instance: bool
        '''
        from numpy import array

        if create_new_instance:
            grid = self.clone()
            grid.origin += array(t)
            return grid

        self.origin += array(t)
        
    def transform(self, R, t):
        '''Returns a new Grid instance that is rotated by R and translated by t.

        @param R: a 3x3 rotation matrix
        @type R: numpy.array

        @param t: a 3-element translation vector (in Angstroms).
        @type t: iterable
        '''
        from numpy import array
        grid = self.rotate(R).prune()
        grid.origin += array(t)
        return grid

    def inertia_tensor(self, return_com = False):

        from numpy import dot, zeros

        com = self.center_of_mass()

        x,y,z = [self.axis(i) - com[i] for i in range(3)]

        I = 1. * zeros((3,3))

        q = dot(x,dot(self.project2D(2),y))

        I[0,1] = q
        I[1,0] = q
        
        q = dot(x,dot(self.project2D(1),z))

        I[0,2] = q
        I[2,0] = q

        q = dot(y,dot(self.project2D(0),z))

        I[1,2] = q
        I[2,1] = q

        I[0,0] = dot(self.project1D(0),x**2)
        I[1,1] = dot(self.project1D(1),y**2)
        I[2,2] = dot(self.project1D(2),z**2)

        if not return_com:
            return I
        else:
            return I, com

    def write_situs(self, filename, transpose_data=True):
        '''Writes the grid to a SITUS format file.

        WARNING: format conversion from situs to e.g. CCP4
        omits origin information. Therefore structure coordinates
        \'X\' only fit the density after \'X -= grid.origin\'

        @param filename: filename for the new SITUS file.
        @type filename: string

        @param transpose_data: If True, the density data is transposed. This was
        used for testing and probably needs to stay True to produce correct
        SITUS maps.
        @type transpose_data: bool
        '''
        import os
        from numpy import array, transpose, reshape, ravel

        #nz, ny, nx = self.nx, self.ny, self.nz
        data = reshape(self.values, (self.nx, self.ny, self.nz))

        if transpose_data:
            data = transpose(data)

        header = '%.6f %.6f %.6f %.6f %i %i %i\n' % \
                 (self.spacing, self.origin[0], self.origin[1], self.origin[2],
                  self.nx, self.ny, self.nz)

        data_string = ''
        entries_this_line = 0
        for i in range(len(data)):
            for j in range (len (data[i])):
                for k in range (len(data[i][j])):

                    if entries_this_line == 0:
                        data_string += '   ' # 3 spaces at the beginning of each data line

                    data_string += '%.6f' % (data[i][j][k]) # the actual data
                    entries_this_line += 1

                    if entries_this_line != 10:
                        data_string += '    ' # 4 spaces between the data entries
                    else:
                        data_string += ' \n' # 1 space and a newline at the end of each line
                        entries_this_line = 0

        f = open(os.path.expanduser(filename), 'w')
        f.write(header + '\n' + data_string) # write header, then one newline, then data
        f.close()

    def prune(self, threshold = 1e-100):
        '''Removes empty slices at both sides of each dimension of the grid.
        
        Note: this works only for positive densities

        @param threshold: Maximal sum of the density values in one slice of the
        grid that will be pruned.
        @type threshold: float
        
        @return: a pruned copy of the grid.
        '''
        from numpy import reshape, sum, ravel, array

        if self.values.min() < 0.:
            raise ValueError('pruning works only for non-negative densities')

        data = reshape(self.values, self.shape)

        ## x direction

        i = 0
        while sum(sum(data[i,:,:])) < threshold:
            i += 1
        I = self.nx-1
        while sum(sum(data[I,:,:])) < threshold:
            I -= 1
        I += 1

        ## y direction

        j = 0
        while sum(sum(data[:,j,:])) < threshold:
            j += 1
        J = self.ny-1
        while sum(sum(data[:,J,:])) < threshold:
            J -= 1
        J += 1

        ## z direction

        k = 0
        while sum(sum(data[:,:,k])) < threshold:
            k += 1
        K = self.nz-1
        while sum(sum(data[:,:,K])) < threshold:
            K -= 1
        K += 1

        data2 = data[i:I,j:J,k:K]
        nx,ny,nz = data2.shape

        grid = Grid(nx,ny,nz, spacing =self.spacing) 
        grid.set_rho(data2)
        origin = array(self.origin) + self.spacing * array([i,j,k])
        grid.origin = tuple(origin.tolist())

        return grid

    def slice(self, R):
        """
        returns a slice through the volume defined by the rotation matrix R
        """
        from scipy.ndimage import map_coordinates
        data = np.reshape(self.values, self.shape)

        x_coords = self.axis(0)
        y_coords = self.axis(1)
        xv, yv = np.meshgrid(x_coords, y_coords)

        slice_coords = np.array([[xx, yy, self.nz/2.]
                                 for xx,yy in zip(xv.ravel(), yv.ravel())])
        slice_coords = np.dot(slice_coords, R)
        grid_coords =  (slice_coords - self.origin)/ self.spacing

        values = map_coordinates(data,
                                 grid_coords.swapaxes(0,1),
                                 order=1, mode='nearest',
                                 prefilter=False)

        return values.reshape((self.nx, self.ny)).T

    def shell_correlation(self, other, binsize=1):
        """
        calculates the cross correlation between two grids 
        over corresponding distances to the center
        
        @param other: other grid with same spacing and origin 
        as self
        @type other: grid
        
        @param nbins: number of bins of the distances
        @type nbins: int
        """
  
        n = [self.nx, self.ny, self.nz]
        x,y,z = [np.linspace(self.origin[i], n[i] * self.spacing, n[i])**2
                 for i in range(3)] 

        indices = {}
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):

                    d = np.sqrt(x[i]+y[j]+z[k])
                    n = int(d/binsize)

                    index = k + self.nz * (j + self.ny * i)
                    if n in indices:
                        indices[n].append(index)
                    else:
                        indices[n] = [index,]

        x = np.arange(max(indices.keys())) * binsize
        fsc = []
        for xx in x:
            try: 
                ix = indices[xx]
                a = self.values[ix] 
                a += 1e-10 * np.random.random(a.shape)
                b = self.values[ix] 
                b += 1e-10 * np.random.random(a.shape)
                if len(a)==1 or len(b)==1:
                    fsc.append(1 - (a[0]-b[0]))
                else:
                    fsc.append(np.corrcoef(a,b)[0])
                
            except KeyError:
                fsc.append(0.0)

        return x, fsc
        

    def resize(self, new_shape):
        '''Returns a new grid object containing the same data as self, but zero
        padded to shape new_shape.

        @param new_shape: the new shape. This must be bigger than or as large as
        the current shape in each dimension, otherwise a ValueError is raised.
        @type new_shape: 3 element vector

        @return: a grid with shape <new_shape>.

        @raise ValueError: If any value in <new_shape> is smaller than its
        correspondance in the grids\' shape.
        '''
        from numpy import Grid, reshape, ravel, array, maximum

        n = array(new_shape)

        if (n == self.shape).all():
            return self

        if (n < array(self.shape)).any():
            raise ValueError('new_shape (%s) must be ' % repr(new_shape) +
                             'at least as large as the current shape ' +
                             '(%s) in each dimension' % repr(self.shape))

        g = Grid(n[0], n[1], n[2], self.spacing)
        g.origin = self.origin

        ## put the old values at the same position in the new Grid
        old_values = reshape(self.values, self.shape)
        new_values = reshape(g.values, new_shape)
        new_values[:self.shape[0], :self.shape[1], :self.shape[2]] = old_values
        g.set_rho(ravel(new_values))
        
        return g

    def cubic(self, side_length=None):
        '''Returns a copy of the grid containing the same data as self, but zero
        padded to cubic shape.

        @param side_length: the target side length of the cube
        @type side_length: int

        @return: a cubic shaped grid with side length <side_length>.

        @raise ValueError: if <side_length> is smaller than any of the grids\'
        dimensions.
        '''
        from numpy import Grid, reshape, ravel, array, maximum

        if side_length is None:
            side_length = array(self.shape).max()

        return self.resize([side_length]*3)
    
    def resample(self, spacing):
        '''Returns a resampled version of the grid.

        @param spacing: the spacing to which the grid is resampled.
        @type spacing: float

        @return: a new grid containing the resampled data.
        '''

        from numpy import ceil, array, identity, zeros

        shape = ceil(array(self.shape) * self.spacing / spacing).astype('i')
        new = Grid(shape[0], shape[1], shape[2], spacing)
        new.origin = tuple(self.origin)

        self.transform_and_interpolate(1.*identity(3), 1.*zeros(3), new.ctype)

        return new

    def normalize(self):
        '''Normalizes the grids\' values to sum to 1.'''
        self.set_rho(self.values/self.values.sum())
    

    def find_shift(self, other, n = 3):
        '''Find optimal shift for superimposing two maps that
        are defined on the same grid
        '''
        from numpy import argmax, ravel, array
        
        cc = other.cc_tensor(self.ctype, n)

        index = argmax(ravel(cc))

        m = 2*n + 1
        i, j, k =  (index / m) / m, (index / m) % m, index % m

        i -= n
        j -= n
        k -= n
        
        if self.debug: print max(ravel(cc))

        return array([i, j, k]) * self.spacing




if __name__ == '__main__':

    import numpy as np
    from csb.bio.io.wwpdb import get
    import scipy
    import scipy.ndimage
    from xfel.io import mrc
    import os

    
    values = mrc.read(os.path.expanduser("~/tmp/test.mrc"))[0]
    values= scipy.ndimage.zoom(values,0.125/2.)
    nx, ny, nz = values.shape
    print nx, ny, nz
    print "creating grid"
    grid = Grid(nx, ny, nz)
    grid.origin = np.array([-nx/2., -ny/2., -nz/2.])

    print "set rho"
    grid.set_rho(values.astype(np.float))
    print "running fsc"
    print grid.shell_correlation(grid, 1.)
    
