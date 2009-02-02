from __future__ import division

import pkg_resources

import numpy as np
import sys, os
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
    my_shlib_fname = os.path.abspath(my_shlib_fname)
    my_shlib = ctypes.CDLL(my_shlib_fname)
else:
    my_shlib_fname = pkg_resources.resource_filename(__name__,'libdumpframe1.so')
    my_shlib_fname = os.path.abspath(my_shlib_fname)
    my_shlib = ctypes.cdll.LoadLibrary(my_shlib_fname)

my_shlib.dumpframe_init.restype = ctypes.c_int
my_shlib.dumpframe_init.argtypes = [ ctypes.c_void_p, # data
                                     ]
my_shlib.dumpframe_close.restype = ctypes.c_int
my_shlib.dumpframe_close.argtypes = [ ctypes.c_void_p, # data
                                     ]

my_shlib.dumpframe_device_init.restype = ctypes.c_int
my_shlib.dumpframe_device_init.argtypes = [ ctypes.c_void_p, # data
                                            ctypes.c_void_p, # data
                                            ]
my_shlib.dumpframe_device_close.restype = ctypes.c_int
my_shlib.dumpframe_device_close.argtypes = [ ctypes.c_void_p, # data
                                             ]

my_shlib.display_frame.restype = ctypes.c_int
my_shlib.display_frame.argtypes = [ ctypes.c_void_p, # data
                                    intptr_type, # stride0
                                    intptr_type, intptr_type, # shape0, shape1
                                    intptr_type, intptr_type, # offset0, offset1
                                    ]

def CHK(retval):
    if retval != 0:
        raise RuntimeError("C callback returned error: %d"%retval)

######################
# initialize
class ModuleT(ctypes.Structure):
    pass
class DeviceT(ctypes.Structure):
    pass
ModuleT_p = ctypes.POINTER(ModuleT)
DeviceT_p = ctypes.POINTER(DeviceT)

global module_ptr
module_ptr = ModuleT_p()

CHK(my_shlib.dumpframe_init(ctypes.byref(module_ptr)))

######################

class DumpframeDevice:
    def __init__(self):
        global module_ptr

        self.device_ptr = DeviceT_p()
        CHK(my_shlib.dumpframe_device_init(module_ptr,ctypes.byref(self.device_ptr)))

    def display_frame( self, arr ):
        arr = np.asarray( arr )
        if arr.shape != ( 4*8, 11*8 ):
            raise ValueError('expected shape of 4x11')
        if arr.dtype != np.uint8:
            raise ValueError('expected uint8 dtype')

        x_offset = y_offset = 0
        CHK( my_shlib.display_frame( self.device_ptr,
                                     arr.ctypes.data,
                                     arr.ctypes.strides[0],
                                     arr.ctypes.shape[0],
                                     arr.ctypes.shape[1],
                                     x_offset,y_offset) )
