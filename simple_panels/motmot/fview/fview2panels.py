from __future__ import division

import pkg_resources

import wx
import numpy
import wx.xrc as xrc
import sys
import Queue
import threading
import time
import pkg_resources
import ctypes

RESFILE = pkg_resources.resource_filename(__name__,"fview2panels.xrc") 
# trigger extraction
RES = xrc.EmptyXmlResource()
RES.LoadFromString(open(RESFILE).read())

######################

# modified from numpy:
def _getintp_ctype():
    char = numpy.dtype('p').char
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

my_shlib.say_hello.restype = None
my_shlib.say_hello.argtypes = [ ]

######################

class PanelDisplayThread( threading.Thread ):
    def __init__(self,
                 queue=None,
                ):
        threading.Thread.__init__(self,name='PanelDisplayThread')
        self._queue = queue

    def _get_most_recent_frame(self):
        buf = None
        quit_now = False
        
        # get the first frame (OK to block - we don't have a frame yet)
        queue_packet = self._queue.get()
        cmd = queue_packet[0]
        if cmd == 'quit':
            quit_now = True
        elif cmd == 'frame':
            buf = queue_packet[1]
            frame_info = queue_packet[2]
        else:
            raise ValueError( 'unknown command' )

        # now, check for any more frames
        while 1:
            try:
                queue_packet = self._queue.get_nowait()
                cmd = queue_packet[0]
                if cmd == 'quit':
                    quit_now = True
                elif cmd == 'frame':
                    buf = queue_packet[1]
                    frame_info = queue_packet[2]
                else:
                    raise ValueError( 'unknown command' )
                
            except Queue.Empty:
                break
        return buf, frame_info, quit_now

    def run(self):
        while 1:
            buf, frame_info, quit_now = self._get_most_recent_frame()
            x_offset, y_offset, timestamp, framenumber = frame_info
            if quit_now:
                break

            # here's where the meat goes...
            
            sys.stdout.write('x')
            sys.stdout.flush()

            if 1:
                retval = my_shlib.display_frame( buf.ctypes.data,
                                                 buf.ctypes.strides[0],
                                                 buf.ctypes.shape[0], buf.ctypes.shape[1],
                                                 x_offset,y_offset)
                                            
                if retval != 0:
                    raise RuntimeError("C callback returned error: %d"%retval)
            time.sleep( 0.100 )

class FView2Panels_Class:
    def __init__(self,wx_parent):
        self.wx_parent = wx_parent
        self.frame = RES.LoadFrame(self.wx_parent,"FVIEW_2_PANELS_FRAME") # make frame main panel
        self._queue = Queue.Queue()

        print 'saying hello from C...'
        my_shlib.say_hello()
        print 'done saying hello from C.'
        
        display_thread = PanelDisplayThread( self._queue )
        display_thread.setDaemon(True) # don't allow the new thread to keep the app alive
        display_thread.start()

    def get_frame(self):
        """return wxPython frame widget"""
        return self.frame

    def get_plugin_name(self):
        return 'FView to USB panels'

    def process_frame(self, cam_id, buf, buf_offset, timestamp, framenumber):
        """do work on each frame

        This function gets called on every single frame capture. It is
        called within the realtime thread, NOT the wxPython
        application mainloop's thread. Therefore, be extremely careful
        (use threading locks) when sharing data with the rest of the
        class.

        """
        
        sys.stdout.write('.')
        sys.stdout.flush()
        
        buf_copy = numpy.array(buf,copy=True)
        x_offset, y_offset = buf_offset
        frame_info = x_offset, y_offset, timestamp, framenumber
        
        self._queue.put( ('frame', buf_copy, frame_info ) )
        
        draw_points = [] #  [ (x,y) ]
        draw_linesegs = [] # [ (x0,y0,x1,y1) ]
        return draw_points, draw_linesegs

    def set_view_flip_LR( self, val ):
        pass

    def set_view_rotate_180( self, val ):
        pass

    def quit(self):
        self._queue.put( ('quit', ) ) # send length 1 tuple

    def camera_starting_notification(self,cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):
        pass
