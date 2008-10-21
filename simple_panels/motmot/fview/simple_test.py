import pkg_resources
from fview2panels import my_shlib
import numpy

print 'starting'
my_shlib.say_hello()
print 'started'
buf = numpy.array( [[33,2,3,4],
                    [5,6,2,8]],
                   dtype = numpy.uint8 )

x_offset = 0
y_offset = 0
print 'calling display frame'
retval = my_shlib.display_frame( buf.ctypes.data,
                                 buf.ctypes.strides[0],
                                 buf.ctypes.shape[0], buf.ctypes.shape[1],
                                 x_offset,y_offset)

print
if retval != 0:
    raise RuntimeError("C callback returned error: %d"%retval)
