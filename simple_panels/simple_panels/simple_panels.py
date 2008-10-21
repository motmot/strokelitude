from __future__ import division

import pkg_resources

import numpy as np
import sys
import pkg_resources
import ctypes

######################

# modified from numpy:
def _getintp_ctype():
    char = np.dtype('p').char
    if (char == 'i'):
        val = ctypes.c_int
    elif char == 'l':
        val = ctypes.c_long
    elif char == 'q':
        val = ctypes.c_longlong
    else:
        raise ValueError, "confused about intp->ctypes."
    _getintp_ctype.cache = val
    return val
intptr_type = _getintp_ctype()

######################

if sys.platform.startswith('win'):
    my_shlib_fname = pkg_resources.resource_filename(__name__,'dumpframe1.dll')
    my_shlib = ctypes.CDLL(my_shlib_fname)
else:
    my_shlib_fname = pkg_resources.resource_filename(__name__,'libdumpframe1.so')
    my_shlib = ctypes.cdll.LoadLibrary(my_shlib_fname)

my_shlib.display_frame.restype = ctypes.c_int
my_shlib.display_frame.argtypes = [ ctypes.c_void_p, # data
                                    intptr_type, # stride0
                                    intptr_type, intptr_type, # shape0, shape1
                                    intptr_type, intptr_type, # offset0, offset1
                                    ]

######################

def display_frame( arr ):
    arr = np.asarray( arr )
    if arr.shape != ( 4*8, 11*8 ):
        raise ValueError('expected shape of 4x11')
    if arr.dtype != np.uint8:
        raise ValueError('expected uint8 dtype')

    x_offset = y_offset = 0
    retval = my_shlib.display_frame( arr.ctypes.data,
                                     arr.ctypes.strides[0],
                                     arr.ctypes.shape[0],
                                     arr.ctypes.shape[1],
                                     x_offset,y_offset)
    if retval != 0:
        raise RuntimeError("C callback returned error: %d"%retval)
