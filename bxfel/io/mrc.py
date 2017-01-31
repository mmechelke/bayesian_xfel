def read_header_raw(filename):

    import os, struct

    mapfile = open(os.path.expanduser(filename))
    header_data = mapfile.read(1024)
    mapfile.close()

    return header_data

def read_header(filename):

    import os, struct

    header_data = read_header_raw(filename)

    header = list(struct.unpack('=10l6f3l3f3l', header_data[:4*25]))
    header.reverse()

    shape = tuple([header.pop() for i in range(3)])
    mode  = header.pop()
    start = tuple([header.pop() for i in range(3)])
    num   = tuple([header.pop() for i in range(3)])
    cella = tuple([header.pop() for i in range(3)])
    cellb = tuple([header.pop() for i in range(3)])
    map   = tuple([header.pop() for i in range(3)])
    stats = tuple([header.pop() for i in range(3)])

    spacing = tuple([cella[i]/num[i] for i in range(3) if num[i]])

    junk = list(struct.unpack('=24l', header_data[4*25:4*(50-1)]))

    origin = list(struct.unpack('=3f4s1l1f1l', header_data[4*(50-1):4*(57-1)]))
    comment = list(struct.unpack('=796s', header_data[4*57:]))

    return shape, spacing, origin[:3]

def read(filename):

    import os, struct

    mapfile = open(os.path.expanduser(filename))

    header_data = mapfile.read(1024)

    NC, NR, NS, MODE, NCSTART, NRSTART, NSSTART, NX, NY, NZ, X, Y, Z, \
        ALPHA, BETA, GAMMA, MAPC, MAPR, MAPS, AMIN, AMAX, AMEAN, \
        ISPG, NSYMBT, LSKFLG = struct.unpack('=10l6f3l3f3l',
                                             header_data[:4*25])
    if MODE == 2 or MODE == 1:
        byte_order = '='

    elif MODE == 33554432:

        NC, NR, NS, MODE, NCSTART, NRSTART, NSSTART, NX, NY, NZ, X, Y, Z, \
            ALPHA, BETA, GAMMA, MAPC, MAPR, MAPS, AMIN, AMAX, AMEAN, \
            ISPG, NSYMBT, LSKFLG = struct.unpack('>10l6f3l3f3l',
                                                 header_data[:4*25])
        byte_order = '>'

        if MODE == 33554432:

            NC, NR, NS, MODE, NCSTART, NRSTART, NSSTART, NX, NY, NZ, \
                X, Y, Z, ALPHA, BETA, GAMMA, MAPC, MAPR, MAPS, \
                AMIN, AMAX, AMEAN, ISPG, NSYMBT, LSKFLG \
                = struct.unpack('<10l6f3l3f3l', header_data[:4*25])
            byte_order = '<'

    else:
        raise IOError("Not a mode 2 CCP4 map file")

    symmetry_data = mapfile.read(NSYMBT)
    map_data = mapfile.read(4*NS*NR*NC)

    import numpy as N
    ## print len(map_data), NC*NR*NS, len(map_data)/float(NC*NR*NS)

    if byte_order == '=':
        array = N.fromstring(map_data, N.float32, NC*NR*NS)
    else:
        array = N.zeros((NS*NR*NC,), N.float32)
        index = 0
        while len(map_data) >= 4*10000:
            values = struct.unpack(byte_order + '10000f',
                                   map_data[:4*10000])
            array[index:index+10000] = N.array(values, N.float32)
            index += 10000
            map_data = map_data[4*10000:]
        values = struct.unpack(byte_order + '%df' % (len(map_data)/4),
                               map_data)
        array[index:] = N.array(values, N.float32)

    del map_data

    array.shape = (NS, NR, NC)
    data = array.T

    spacing = [0,0,0]
    if NX != 0 and NY != 0 and NZ != 0:
        spacing = X/NX, Y/NY, Z/NZ
    origin  = NCSTART, NRSTART, NSSTART
    origin  = [origin[i] * spacing[i] for i in range(3)]

    return data, spacing, origin

def write(data, filename, spacing=1.0, origin=None, comments=None,
          axes = None):

    import struct, os
    from numpy import argsort, take, float32

    N = list(data.shape)

    MODE = 2 ## byteorder = '='

    if type(spacing) == float:
        spacing = 3 * [spacing]
    if origin is None:
        origin = 3 * [0.]

    start = [int(round(origin[i] / spacing[i],0)) for i in range(3)]

    M = [data.shape[i]-1 for i in range(3)]

    cella = [(data.shape[i]-1) * spacing[i] for i in range(3)]
    cellb = 3 * [90.]

    if axes is None:
        MAP = range(1,4)
    else:
        MAP = list(axes)

    stats = [data.min(), data.max(), data.mean()]

    ISPG = 1L
    NSYMBT = 0
    LSKFLG = 0

    JUNK = [0] * 25
    ORIGIN = [0.,0.,0.]
    MACHST = 0

    args = N + [MODE] + start + M + cella + cellb + \
           MAP + stats + [ISPG, NSYMBT, LSKFLG] + JUNK + \
           ORIGIN + [0, MACHST, 0., 0] + [' ' * 796]

    f = open(os.path.expanduser(filename),'wb')
    x = struct.pack('=10l6f3l3f3l25l3f2l1f1l796s',*args)
    f.write(x)
    data = data.T

    #x = ''.join(data.astype(float32).data)
    data = data.flatten()
    x = struct.pack('=%df' % len(data), *data.tolist())
    f.write(x)
    f.close()


def write_sparse_data(data, filename, spacing=1.0, origin=None, comments=None,
                      axes = None):

    x = data.toarray()
    x.swapaxes(0,2)
    
def write_data(data, filename, header):

    import os, struct

    if len(header) != 1024:
        header = read_header_raw(header)

    f = open(os.path.expanduser(filename),'w')
    f.write(header)

    data = data.T.flatten()
    x = struct.pack('=%df' % len(data), *data.tolist())
    f.write(x)
    f.close()

