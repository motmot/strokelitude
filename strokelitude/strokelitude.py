from __future__ import division, with_statement

import pkg_resources
import data_format # strokelitude

import motmot.utils.config
import motmot.FastImage.FastImage as FastImage
import motmot.realtime_image_analysis.realtime_image_analysis as realtime_image_analysis
import motmot.fview.traited_plugin as traited_plugin

# Weird bugs arise in class identity testing if this is not absolute
#   import...  ... but Python won't let me import it as absolute. Ahh,
#   well, workaround the bug by setting selectable on InstanceEditor
#   directly.
import plugin as strokelitude_plugin_module

import wx
import warnings, time, threading

import numpy as np
import scipy.ndimage
import cairo
import Queue
import os, warnings, pickle
import motmot.FlyMovieFormat.FlyMovieFormat as FMF

import tables # pytables

import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor, InstanceEditor, ButtonEditor

from enthought.enable.api import Component, Container
from enthought.enable.wx_backend.api import Window
from enthought.chaco.api import DataView, ArrayDataSource, ScatterPlot, \
     LinePlot, LinearMapper, ArrayPlotData, Plot, gray
from enthought.enable.component_editor import ComponentEditor
import enthought.chaco.default_colormaps as default_colormaps
from enthought.chaco.data_range_1d import DataRange1D
from enthought.chaco.api import create_line_plot, add_default_axes, \
     add_default_grids
from enthought.chaco.tools.api import PanTool, ZoomTool
from enthought.chaco.tools.image_inspector_tool import ImageInspectorTool, \
     ImageInspectorOverlay
import motmot.fview_ext_trig.live_timestamp_modeler as modeler_module

import math
import time


# ROS stuff ------------------------------
try:
    import roslib
    from roslib.packages import InvalidROSPkgException
    have_ROS = True
except ImportError, err:
    have_ROS = False

if have_ROS:
    try:
        roslib.load_manifest('strokelitude_ros')
    except InvalidROSPkgException:
        have_ROS = False
    else:
        from strokelitude_ros.msg import FlyWingEstimate
        import rospy
        import rospy.core
# -----------------------------------------

DataReadyEvent = wx.NewEventType()
BGReadyEvent = wx.NewEventType()

D2R = np.pi/180.0
R2D = 180.0/np.pi

def load_plugins():
    # modified from motmot.fview.plugin_manager
    PluginClasses = []
    pkg_env = pkg_resources.Environment()
    for name in pkg_env:
        egg = pkg_env[name][0]
        modules = []

        for name in egg.get_entry_map('strokelitude.plugins'):
            egg.activate()
            entry_point = egg.get_entry_info('strokelitude.plugins', name)
            try:
                PluginClass = entry_point.load()
            except Exception,x:
                if int(os.environ.get('FVIEW_RAISE_ERRORS','0')):
                    raise
                else:
                    import warnings
                    warnings.warn('could not load plugin (set env var '
                                  'FVIEW_RAISE_ERRORS to raise error) %s: %s'%(
                        str(entry_point),str(x)))
                    continue
            PluginClasses.append( PluginClass )
            modules.append(entry_point.module_name)
    # make instances of plugins
    plugins = [PluginClass() for PluginClass in PluginClasses]
    return plugins

def compute_hough_lookup_table(x0,y0,gamma,sign,width,height,frac=1):
    x = (np.arange(width)[::frac]-x0)
    y = (np.arange(height)[::frac]-y0)[:,np.newaxis]
    angle_image_coords = np.arctan2(y,x)
    angle_wing_coords = sign*(angle_image_coords - gamma) - np.pi/2 # pi/2 is to shift interesting range from 90-270
    angle_wing_coords = np.mod(angle_wing_coords,2*np.pi) # 0 <= result < 2*pi
    return angle_wing_coords

class BufferAllocator:
    def __call__(self, w, h):
        return FastImage.FastImage8u(FastImage.Size(w,h))

mult_sign={True:{'left':1,
                 'right':-1},
           False:{'left':-1,
                  'right':1}}

class MaskData(traits.HasTraits):
    # lengths, in pixels

    # The upper bounds of these ranges should all be dynamically set
    # based on image size
    x = traits.Range(0.0, 656.0, 401.0, mode='slider', set_enter=True)
    y = traits.Range(0.0, 491.0, 273.5, mode='slider', set_enter=True)
    wingsplit = traits.Range(0.0, 180.0, 70.3, mode='slider', set_enter=True)

    view_from_below = traits.Bool(True)

    # The bounds of these should be dynamically set
    r1 = traits.Range(0.0, 656.0, 22.0, mode='slider', set_enter=True)
    r2 = traits.Range(0.0, 656.0, 22.0, mode='slider', set_enter=True)

    # angles, in degrees
    alpha = traits.Range(-90.0, 90.0, 87.0, mode='slider', set_enter=True)
    beta = traits.Range(-90.0, 90.0, 14.0, mode='slider', set_enter=True)
    gamma = traits.Range(0.0, 360.0, 206.0, mode='slider', set_enter=True)

    # number of angular bins
    nbins = traits.Int(30)

    # these are just necesary for establishing limits in the view:
    maxx = traits.Float(699.9)
    maxy = traits.Float(699.9)
    maxdim = traits.Property(depends_on=['maxx','maxy'])

    rotation = traits.Property(depends_on=['gamma'])
    translation = traits.Property(depends_on=['x','y'])

    quads_left = traits.Property(depends_on=[
        'wingsplit', 'r1', 'r2', 'alpha', 'beta', 'nbins',
        'rotation','translation','view_from_below',
        ])

    left_mat = traits.Property(depends_on=['quads_left'])
    left_mat_half = traits.Property(depends_on=['quads_left'])
    left_mat_quarter = traits.Property(depends_on=['quads_left'])
    left_mat_quarter_norm = traits.Property(depends_on=['left_mat_quarter'])

    mean_quad_angles_deg = traits.Property(depends_on=[
        'alpha', 'beta', 'nbins'])

    quads_right = traits.Property(depends_on=[
        'wingsplit', 'r1', 'r2', 'alpha', 'beta', 'nbins',
        'rotation','translation','view_from_below',
        ])
    right_mat = traits.Property(depends_on=['quads_right'])
    right_mat_half = traits.Property(depends_on=['quads_right'])
    right_mat_quarter = traits.Property(depends_on=['quads_right'])
    right_mat_quarter_norm = traits.Property(depends_on=['right_mat_quarter'])

    extra_linesegs = traits.Property(depends_on=[
        'wingsplit', 'r1', 'r2', 'alpha', 'beta', 'nbins',
        'rotation','translation','view_from_below',
        ])

    all_linesegs = traits.Property(depends_on=[
        'quads_left', 'quads_right', 'extra_linesegs',
        ] )

    @traits.cached_property
    def _get_all_linesegs(self):
        # concatenate lists
        return (self.quads_left + self.quads_right + self.extra_linesegs)

    @traits.cached_property
    def _get_maxdim(self):
        return np.sqrt(self.maxx**2+self.maxy**2)

    def _get_wingsplit_translation(self,side):
        sign = mult_sign[self.view_from_below][side]
        return np.array([[0.0],[sign*self.wingsplit]])

    @traits.cached_property
    def _get_rotation(self):
        gamma = self.gamma*D2R
        return np.array([[ np.cos( gamma ), -np.sin(gamma)],
                         [ np.sin( gamma ), np.cos(gamma)]])

    @traits.cached_property
    def _get_translation(self):
        return np.array( [[self.x],
                          [self.y]],
                         dtype=np.float64 )

    traits_view = View( Group( ( Item('x',
                                      style='custom',
                                      ),
                                 Item('y',
                                      style='custom',
                                      ),
                                 Item('view_from_below'),
                                 Item('wingsplit',
                                      style='custom',
                                      ),
                                 Item('r1',style='custom',
                                      ),
                                 Item('r2',style='custom',
                                      ),
                                 Item('alpha',style='custom'),
                                 Item('beta',style='custom'),
                                 Item('gamma',style='custom'),
                                 ),
                               ),
                        title = 'Mask Parameters',
                        resizable = True,
                        )

    @traits.cached_property
    def _get_extra_linesegs(self):
        return self.get_extra_linesegs()

    @traits.cached_property
    def _get_quads_left(self):
        return self.get_quads('left')

    @traits.cached_property
    def _get_quads_right(self):
        return self.get_quads('right')

    @traits.cached_property
    def _get_left_mat(self):
        result = []
        for quad in self.quads_left:
            fi_roi, left, bottom = quad2fastimage_offset(quad,
                                                         self.maxx,self.maxy)
            result.append( (fi_roi, left, bottom) )
        return result

    @traits.cached_property
    def _get_left_mat_half(self):
        result = []
        for quad in self.quads_left:
            fi_roi, left, bottom = quad2fastimage_offset(quad,
                                                         self.maxx,self.maxy,
                                                         frac=2)
            result.append( (fi_roi, left, bottom) )
        return result

    @traits.cached_property
    def _get_left_mat_quarter(self):
        result = []
        for quad in self.quads_left:
            fi_roi, left, bottom = quad2fastimage_offset(quad,
                                                         self.maxx,self.maxy,
                                                         frac=4)
            result.append( (fi_roi, left, bottom) )
        return result

    @traits.cached_property
    def _get_left_mat_quarter_norm(self):
        result = []
        for (fi_roi, left, bottom) in self.left_mat_quarter:
            result.append( 1.0/np.sum(fi_roi) )
        return np.array(result)

    @traits.cached_property
    def _get_right_mat(self):
        result = []
        for quad in self.quads_right:
            fi_roi, left, bottom = quad2fastimage_offset(quad,
                                                         self.maxx,self.maxy)
            result.append( (fi_roi, left, bottom) )
        return result

    @traits.cached_property
    def _get_right_mat_half(self):
        result = []
        for quad in self.quads_right:
            fi_roi, left, bottom = quad2fastimage_offset(quad,
                                                         self.maxx,self.maxy,
                                                         frac=2)
            result.append( (fi_roi, left, bottom) )
        return result

    @traits.cached_property
    def _get_right_mat_quarter(self):
        result = []
        for quad in self.quads_right:
            fi_roi, left, bottom = quad2fastimage_offset(quad,
                                                         self.maxx,self.maxy,
                                                         frac=4)
            result.append( (fi_roi, left, bottom) )
        return result

    @traits.cached_property
    def _get_right_mat_quarter_norm(self):
        result = []
        for (fi_roi, right, bottom) in self.right_mat_quarter:
            result.append( 1.0/np.sum(fi_roi) )
        return np.array(result)

    @traits.cached_property
    def _get_mean_quad_angles_deg(self):
        x = np.linspace(self.alpha,self.beta,self.nbins+1)
        av = (x[:-1]+x[1:])*0.5 # local means
        return av

    def get_extra_linesegs(self):
        """return linesegments that contextualize parameters"""
        linesegs = []
        if 1:
            # longitudinal axis (along fly's x coord)
            verts = np.array([[-100,100],
                              [0,     0]],dtype=np.float)
            verts = np.dot(self.rotation, verts) + self.translation
            linesegs.append( verts.T.ravel() )
        if 1:
            # transverse axis (along fly's y coord)
            verts = np.array([[0,0],
                              [-10,10]],dtype=np.float)
            verts = np.dot(self.rotation, verts) + self.translation
            linesegs.append( verts.T.ravel() )
        if 1:
            # half-circle centers
            n = 10
            theta = np.linspace(0,2*np.pi,10)
            verts = np.array([ 5*np.cos(theta),
                               5*np.sin(theta) ])
            vleft = verts+self._get_wingsplit_translation('left')
            vright = verts+self._get_wingsplit_translation('right')

            vleft = np.dot(self.rotation, vleft) + self.translation
            vright = np.dot(self.rotation, vright) + self.translation

            linesegs.append( vleft.T.ravel() )
            linesegs.append( vright.T.ravel() )

        if 1:
            # left wing
            verts = np.array([[0.0, 0.0, 20.0],
                              [20.0,0.0, 0.0]])
            vleft = verts+self._get_wingsplit_translation('left')

            vleft = np.dot(self.rotation, vleft) + self.translation

            linesegs.append( vleft.T.ravel() )

        return linesegs

    def get_quads(self,side):
        """return linesegments outlining the pattern on a given side.

        This can be used to draw the pattern (e.g. wtih OpenGL).

        Return a list of linesegments [seg1, seg2, ..., segn]

        Each line segment is in the form (x0,y0, x1,y1, ..., xn,yn)

        """
        # This work is insignificant compared to the function call
        # overhead to draw the linesegs in the dumb way done by
        # wxglvideo (caching the values didn't reduce CPU time on
        # "Mesa DRI Intel(R) 946GZ 4.1.3002 x86/MMX/SSE2, OpenGL 1.4
        # Mesa 7.0.3-rc2") - ADS 20081015

        alpha = self.alpha*D2R
        beta = self.beta*D2R

        sign = mult_sign[self.view_from_below][side]

        # ordered from front to back, zero straight out from fly, positive towards head
        all_theta_user_coords = np.linspace(alpha,beta,self.nbins+1)

        # convert to drawing coord system
        all_theta = 90*D2R - all_theta_user_coords # make
        all_theta *= sign

        linesegs = []
        for i in range(self.nbins):
            theta = all_theta[i:(i+2)]
            # inner radius
            inner = np.array([self.r1*np.cos(theta),
                              self.r1*np.sin(theta)])
            # outer radius
            outer = np.array([self.r2*np.cos(theta[::-1]),
                              self.r2*np.sin(theta[::-1])])

            wing_verts =  np.hstack(( inner, outer, inner[:,np.newaxis,0] ))
            wing_verts += self._get_wingsplit_translation(side)

            wing_verts = np.dot(self.rotation, wing_verts) + self.translation
            linesegs.append( wing_verts.T.ravel() )

        return linesegs

    def index2angle(self,side,idx):
        """convert index to angle (in radians)"""
        alpha = self.alpha*D2R
        beta = self.beta*D2R

        frac = idx/self.nbins
        diff = beta-alpha
        return alpha+frac*diff

    def get_span_lineseg(self,side,theta_user):
        """draw line on side at angle theta_user (in radians)"""
        linesegs = []
        sign = mult_sign[self.view_from_below][side]
        theta = 90*D2R - theta_user
        theta *= sign

        verts = np.array( [[ 0, 1000.0*np.cos(theta)],
                           [ 0, 1000.0*np.sin(theta)]] )
        verts = verts + self._get_wingsplit_translation(side)
        verts = np.dot(self.rotation, verts) + self.translation
        linesegs.append( verts.T.ravel() )
        return linesegs

def quad2fastimage_offset(quad,width,height,debug_count=0,frac=1,float32=False):
    """convert a quad to an image vector"""
    mult = 1.0/frac
    newwidth = width//frac
    newheight = height//frac
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                 width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0,0,0)

    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.paint()

    x0=quad[0]
    y0=quad[1]
    x0 = x0*mult
    y0 = y0*mult
    ctx.move_to(x0,y0)
    xmin = int(np.floor(x0))
    xmax = int(np.ceil(x0+1))
    ymin = int(np.floor(y0))
    ymax = int(np.ceil(y0+1))

    xs = quad[2::2]
    ys = quad[3::2]
    xs = xs*mult
    ys = ys*mult
    for (x,y) in zip(xs,ys):
        ctx.line_to(x,y)
        xmin = min(int(np.floor(x)),xmin)
        xmax = max(int(np.ceil(x+1)),xmax)
        ymin = min(int(np.floor(y)),ymin)
        ymax = max(int(np.ceil(y+1)),ymax)
    ctx.close_path()
    ctx.set_source_rgb(1,1,1)
    ctx.fill()
    if 0:
        fname = 'poly_left_%05d.png'%debug_count
        print 'saving',fname
        surface.write_to_png(fname)
    buf = surface.get_data()

    # Now, convert to numpy
    arr = np.frombuffer(buf, np.uint8)
    arr.shape = (height, width, 4)
    if 0:
        import scipy.misc.pilutil
        fname = 'mask_%05d.png'%debug_count
        print 'saving',fname
        im = scipy.misc.pilutil.toimage(arr)
        im.save(fname)
    arr = arr[:,:,0] # only red channel
    if float32:
        arr = arr.astype(np.float32)
    else:
        # uint8
        arr = np.array(arr,copy=True)
    # Now, crop to only region of interest
    roi = arr[ymin:ymax,xmin:xmax]
    fi_roi = FastImage.asfastimage(roi)
    offset = xmin,ymin
    return fi_roi, xmin, ymin

def compute_sparse_mult(list_of_fi_offsets, fi_im):
    result = []
    for imA,left,bottom in list_of_fi_offsets:
        imB = fi_im.roi( left, bottom, imA.size )
        result.append( imA.dot(imB, imA.size) )
    result = np.array(result)
    return result

StrokelitudeDataDescription = data_format.StrokelitudeDataDescription
StrokelitudeDataCol_dtype = tables.Description(
    StrokelitudeDataDescription().columns)._v_nestedDescr

AnalogInputWordstreamDescription = data_format.AnalogInputWordstreamDescription
AnalogInputWordstream_dtype =  tables.Description(
    AnalogInputWordstreamDescription().columns)._v_nestedDescr

TimeDataDescription = data_format.TimeDataDescription
TimeData_dtype =  tables.Description(
    TimeDataDescription().columns)._v_nestedDescr

class LiveDataPlot(traits.HasTraits):
    left_angle = traits.Float
    right_angle = traits.Float
    left_plot = traits.Instance(Plot)
    right_plot = traits.Instance(Plot)
    traits_view = View(Group(Item('left_angle',style='readonly'),
                             Item('right_angle',style='readonly'),
                             Item('left_plot',
                                  editor=ComponentEditor(),
                                  height=300,
                                  width=500,
                                  show_label=False,
                                  ),
                             Item('right_plot',
                                  editor=ComponentEditor(),
                                  height=300,
                                  width=500,
                                  show_label=False,
                                  ),
                             ),
                       title='live data plot',
                       resizable = True,
                       )

class PopUpPlot(traits.HasTraits):
    plot = traits.Instance(Plot)
    traits_view = View(Group(
                             Item('plot',
                                  editor=ComponentEditor(),
                                  height=300,
                                  width=500,
                                  show_label=False,
                                  ),
                             ),
                       title='plot',
                       resizable = True,
                       )

class AmplitudeFinder(traits.HasTraits):
    strokelitude_instance = traits.Any # really, instance of StrokelitudeClass

    def post_init(self, strokelitude_instance):
        self.strokelitude_instance = strokelitude_instance # my parent

    def camera_starting_notification(self,cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):
        # this method exists primarily to be overriden
        pass

    def offline_startup_func(self,arg):
        # this method exists primarily to be overriden
        pass

    def process_frame(self,buf,buf_offset,timestamp,framenumber):
        """do the work"""
        left_angle_degrees = np.nan
        right_angle_degrees = np.nan

        return left_angle_degrees, right_angle_degrees

class BackgroundSubtractionDotProductFinder(AmplitudeFinder):
    mask_dirty = traits.Bool(False) # let __init__ set True

    recompute_mask = traits.Event
    take_bg = traits.Event
    clear_bg = traits.Event

    background_viewer = traits.Instance(PopUpPlot)
    live_data_viewer = traits.Instance(LiveDataPlot)
    live_data_plot = traits.Instance(LiveDataPlot)

    processing_enabled = traits.Bool(False)
    light_on_dark = traits.Bool(True)
    threshold_fraction = traits.Float(0.5)

    flight_detection_enabled = traits.Bool(True)
    flight_detection_threshold = traits.Float(1e6)

    traits_view = View( Group( Item('recompute_mask',
                                    editor=ButtonEditor(),show_label=False),
                               Item('processing_enabled'),
                               Item('take_bg',
                                    editor=ButtonEditor(),show_label=False),
                               Item('clear_bg',
                                    editor=ButtonEditor(),show_label=False),
                               Item('background_viewer',show_label=False),
                               Item('live_data_viewer',show_label=False),
                               Item('live_data_plot',show_label=False),
                               Item(name='threshold_fraction',
                                    ),
                               Item(name='light_on_dark',
                                    ),
                               Item(name='flight_detection_enabled',
                                    ),
                               Item(name='flight_detection_threshold',
                                    ),
                               ))
    def __init__(self,*args,**kwargs):
        super(BackgroundSubtractionDotProductFinder,self).__init__(*args,**kwargs)
        self.bg_cmd_queue = Queue.Queue()
        self.new_bg_image_lock = threading.Lock()
        self.new_bg_image = None
        self.recomputing_lock = threading.Lock()

        self.background_viewer = PopUpPlot()
        self.live_data_plot = LiveDataPlot()
        self.vals_queue = Queue.Queue()

    def _take_bg_fired(self):
        self.mask_dirty = True
        self.bg_cmd_queue.put('take')

    def _clear_bg_fired(self):
        self.mask_dirty = True
        self.bg_cmd_queue.put('clear')

    def OnNewBGReady(self,event):
        with self.new_bg_image_lock:
            new_bg_image = self.new_bg_image
            self.new_bg_image = None
        if self.strokelitude_instance.save_to_disk:
            print 'should save BG image to disk'
        self.bg_pd.set_data("imagedata", new_bg_image)
        self.bg_plot.request_redraw()
        # we have not yet done the sparse multiplication yet

    def OnDataReady(self, event):
        lrvals = None
        while 1:
            # We only care about most recent results. Discard older data.
            try:
                lrvals = self.vals_queue.get_nowait()
            except Queue.Empty:
                break
        if lrvals is None:
            return

        left_vals, right_vals,left_angle_degrees,right_angle_degrees = lrvals
        self.live_pd['left'].set_data('live',left_vals)
        self.live_pd['right'].set_data('live',right_vals)
        self.live_data_plot.left_angle = left_angle_degrees
        self.live_data_plot.right_angle = right_angle_degrees

    def post_init(self, *args,**kwargs):
        super(BackgroundSubtractionDotProductFinder,self).post_init(*args,**kwargs)

        self.strokelitude_instance.frame.Connect( -1, -1, DataReadyEvent, self.OnDataReady )
        self.strokelitude_instance.frame.Connect( -1, -1, BGReadyEvent, self.OnNewBGReady )
        self.mask_dirty = True

        if 1:
            self.live_pd = {}
            for side in ['left','right']:

                x=self.strokelitude_instance.maskdata.mean_quad_angles_deg
                y=np.zeros_like(x)

                pd=ArrayPlotData(index=x)
                pd.set_data('live',y)
                pd.set_data('bg',np.zeros_like(x))
                self.live_pd[side]=pd
                plot = Plot(pd, title=side,
                            padding=50,
                            border_visible=True,
                            overlay_border=True)
                plot.legend.visible = True
                plot.plot(("index", 'live'), name=side, color="red")
                plot.plot(("index", "bg"), name="background", color="blue")

                if side=='left':
                    self.left_plot=plot
                    self.live_data_plot.left_plot=plot
                elif side=='right':
                    self.right_plot=plot
                    self.live_data_plot.right_plot=plot
            del plot,x,y,pd # don't let these escape and pollute namespace

        if 1:
            # a temporary initial background image
            image = np.zeros((640,480), dtype=np.uint8)

            # Create a plot data obect and give it this data
            self.bg_pd = ArrayPlotData()
            self.bg_pd.set_data("imagedata", image)

            # Create the plot
            self.bg_plot = Plot(self.bg_pd, default_origin="top left")
            self.bg_plot.x_axis.orientation = "top"
            colormap = default_colormaps.gray(DataRange1D(low=0,high=255))
            self.bg_plot.img_plot("imagedata",colormap=colormap)[0]

            self.bg_plot.padding = 30
            self.bg_plot.bgcolor = "white"

            # Attach some tools to the plot
            self.bg_plot.tools.append(PanTool(self.bg_plot,
                                              constrain_key="shift"))
            self.bg_plot.overlays.append(ZoomTool(component=self.bg_plot,
                                          tool_mode="box", always_on=False))

            self.background_viewer.plot = self.bg_plot

        self.strokelitude_instance.maskdata.on_trait_change( self.on_mask_change )

    def camera_starting_notification(self,cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):

        self.width = max_width
        self.height = max_height

        with self.recomputing_lock:
            self.bg_image = np.zeros( (self.height,self.width), dtype=np.uint8 )

    def offline_startup_func(self,arg):
        # automatically recompute mask and enable processing in offline mode
        self.recompute_mask = True # fire event
        assert self.mask_dirty==False
        self.processing_enabled = True

    def process_frame(self,buf,buf_offset,timestamp,framenumber):
        """do the work"""
        left_angle_degrees = np.nan
        right_angle_degrees = np.nan

        have_lock = self.recomputing_lock.acquire(False) # don't block
        if not have_lock:
            return left_angle_degrees, right_angle_degrees

        try:
            command = None
            while 1:
                try:
                    # all we care about is last command
                    command = self.bg_cmd_queue.get_nowait()
                except Queue.Empty, err:
                    break

            bg_image_changed = False
            this_image = np.asarray(buf)
            if command == 'clear':
                bg_image_changed = True
                self.bg_image = np.zeros_like( this_image )
            elif command == 'take':
                bg_image_changed = True
                self.bg_image = np.array( this_image, copy=True )
            if bg_image_changed:
                with self.new_bg_image_lock:
                    self.new_bg_image = np.array( self.bg_image, copy=True )

                event = wx.CommandEvent(BGReadyEvent)
                event.SetEventObject(self.strokelitude_instance.frame)

                # trigger call to self.OnNewBGReady
                wx.PostEvent(self.strokelitude_instance.frame, event)

            # XXX naughty to cross thread boundary to get enabled_box value, too
            if (self.processing_enabled and not self.mask_dirty):

                h,w = this_image.shape
                if not (self.width==w and self.height==h):
                    raise NotImplementedError('need to support ROI')

                left_mat = self.left_mat
                right_mat = self.right_mat

                bg_left_vec = self.bg_left_vec
                bg_right_vec = self.bg_right_vec

                this_image_fi = FastImage.asfastimage(this_image)
                left_vals  = compute_sparse_mult(left_mat,this_image_fi)
                right_vals = compute_sparse_mult(right_mat,this_image_fi)

                left_vals = left_vals - bg_left_vec
                right_vals = right_vals - bg_right_vec

                results = []
                for side in ('left','right'):
                    if side=='left':
                        vals = left_vals
                    else:
                        vals = right_vals

                    if not self.light_on_dark:
                        vals = -vals

                    min_val = vals.min()
                    mid_val = (vals.max() - min_val)*self.threshold_fraction + min_val
                    if min_val==mid_val:
                        # no variation in luminance
                        interp_idx = -1
                    else:
                        first_idx=None
                        for i in range(len(vals)-1):
                            if (vals[i] < mid_val) and (vals[i+1] >= mid_val):
                                first_idx = i
                                second_idx = i+1
                                break
                        if first_idx is None:
                            interp_idx = -1
                        else:
                            # slope (indices are unity apart)
                            # y = mx+b
                            m = vals[second_idx] - vals[first_idx]
                            b = vals[first_idx] - m*first_idx
                            interp_idx = (mid_val-b)/m

                    #latency_sec = time.time()-timestamp
                    #print 'msec % 5.1f'%(latency_sec*1000.0,)

                    if interp_idx != -1:
                        angle_radians = self.strokelitude_instance.maskdata.index2angle(side,
                                                                                        interp_idx)
                        results.append( angle_radians*R2D ) # keep results in degrees

                    else:
                        results.append( np.nan )

                left_angle_degrees, right_angle_degrees = results
                del results

                if self.flight_detection_enabled:
                    if ((np.mean(left_vals) <= self.flight_detection_threshold) and
                        (np.mean(right_vals) <= self.flight_detection_threshold)):
                        beta = self.strokelitude_instance.maskdata.beta
                        left_angle_degrees = -90.0
                        right_angle_degrees = -90.0

                ## for queue in self.plugin_data_queues:
                ##     queue.put( (cam_id,timestamp,framenumber,results) )

                # send values from each quad to be drawn
                self.vals_queue.put((left_vals, right_vals,
                                     left_angle_degrees,right_angle_degrees,
                                     ))
                event = wx.CommandEvent(DataReadyEvent)
                event.SetEventObject(self.strokelitude_instance.frame)

                # trigger call to self.OnDataReady
                wx.PostEvent(self.strokelitude_instance.frame, event)

        finally:
            self.recomputing_lock.release()
        return left_angle_degrees, right_angle_degrees

    def _recompute_mask_fired(self):
        with self.recomputing_lock:
            left_quads = self.strokelitude_instance.maskdata.quads_left
            right_quads = self.strokelitude_instance.maskdata.quads_right

            self.left_mat = self.strokelitude_instance.maskdata.left_mat
            self.right_mat = self.strokelitude_instance.maskdata.right_mat

            bg = FastImage.asfastimage(self.bg_image)
            self.bg_left_vec = compute_sparse_mult(self.left_mat,bg)
            self.bg_right_vec= compute_sparse_mult(self.right_mat,bg)

            self.live_pd['left'].set_data('bg',self.bg_left_vec)
            self.live_pd['right'].set_data('bg',self.bg_right_vec)
            #self.left_plot.request_redraw()
            #self.right_plot.request_redraw()

            self.mask_dirty=False

    def on_mask_change(self):
        self.mask_dirty=True

    def _mask_dirty_changed(self):
        if self.mask_dirty:
            self.processing_enabled = False

class SobelFinder(AmplitudeFinder):
    processing_enabled = traits.Bool(False)
    mask_thresh = traits.Int(150)
    maxG = traits.Int(400)
    n_iterations_dilation = traits.Int(10)
    gaussian_sigma = traits.Float(3.0)
    line_strength_thresh = traits.Float(20.0)
    update_plots = traits.Bool(False)

    bg_viewer = traits.Instance(PopUpPlot)
    a1_viewer = traits.Instance(PopUpPlot)
    a2_viewer = traits.Instance(PopUpPlot)
    a3_viewer = traits.Instance(PopUpPlot)

    traits_view = View( Group( Item('processing_enabled'),
                               Item('mask_thresh'),
                               Item('maxG'),
                               Item('n_iterations_dilation'),
                               Item('gaussian_sigma'),
                               Item('line_strength_thresh'),
                               Item('update_plots'),
                               Item('bg_viewer',show_label=False),
                               Item('a1_viewer',show_label=False),
                               Item('a2_viewer',show_label=False),
                               Item('a3_viewer',show_label=False),
                               ))
    def __init__(self,*args,**kwargs):
        super(SobelFinder,self).__init__(*args,**kwargs)
        self.bg_viewer = PopUpPlot()
        if 1:
            # a temporary initial background image
            image = np.zeros((480,640), dtype=np.uint8)

            # Create a plot data obect and give it this data
            self.bg_pd = ArrayPlotData()
            self.bg_pd.set_data("imagedata", image)
            # Create the plot
            bg_plot = Plot(self.bg_pd, default_origin="bottom left")
            bg_plot.x_axis.orientation = "top"
            colormap = default_colormaps.gray(DataRange1D(low=0,high=255))
            bg_plot.img_plot("imagedata",colormap=colormap)[0]

            bg_plot.padding = 30
            bg_plot.bgcolor = "white"

            # Attach some tools to the plot
            bg_plot.tools.append(PanTool(bg_plot,
                                         constrain_key="shift"))
            bg_plot.overlays.append(ZoomTool(component=bg_plot,
                                             tool_mode="box", always_on=False))

            self.bg_viewer.plot = bg_plot
        self.a1_viewer = PopUpPlot()
        if 1:
            # a temporary initial background image
            image = np.zeros((480,640), dtype=np.uint8)

            # Create a plot data obect and give it this data
            self.a1_pd = ArrayPlotData()
            self.a1_pd.set_data("imagedata", image)
            # Create the plot
            a1_plot = Plot(self.a1_pd, default_origin="bottom left")
            a1_plot.x_axis.orientation = "top"
            colormap = default_colormaps.gray(DataRange1D(low=0,high=255))
            a1_plot.img_plot("imagedata",colormap=colormap)[0]

            a1_plot.padding = 30
            a1_plot.bgcolor = "white"

            # Attach some tools to the plot
            a1_plot.tools.append(PanTool(a1_plot,
                                         constrain_key="shift"))
            a1_plot.overlays.append(ZoomTool(component=a1_plot,
                                             tool_mode="box", always_on=False))

            self.a1_viewer.plot = a1_plot
        self.a2_viewer = PopUpPlot()
        if 1:
            # a temporary initial background image
            image = np.zeros((480,640), dtype=np.float)

            # Create a plot data obect and give it this data
            self.a2_pd = ArrayPlotData()
            self.a2_pd.set_data("imagedata", image)
            # Create the plot
            a2_plot = Plot(self.a2_pd, default_origin="bottom left")
            a2_plot.x_axis.orientation = "top"
            colormap = default_colormaps.jet(DataRange1D(low=0,high=255))
            a2_plot.img_plot("imagedata",colormap=colormap)[0]

            a2_plot.padding = 30
            a2_plot.bgcolor = "white"

            # Attach some tools to the plot
            a2_plot.tools.append(PanTool(a2_plot,
                                         constrain_key="shift"))
            a2_plot.overlays.append(ZoomTool(component=a2_plot,
                                             tool_mode="box", always_on=False))

            self.a2_viewer.plot = a2_plot
        self.a3_viewer = PopUpPlot()
        if 1:
            # a temporary initial background image
            image = np.zeros((480,640), dtype=np.float)

            # Create a plot data obect and give it this data
            self.a3_pd = ArrayPlotData()
            self.a3_pd.set_data("imagedata", image)
            # Create the plot
            a3_plot = Plot(self.a3_pd, default_origin="bottom left")
            a3_plot.x_axis.orientation = "top"
            colormap = default_colormaps.jet(DataRange1D(low=0,high=255))
            a3_plot.img_plot("imagedata",colormap=colormap)[0]

            a3_plot.padding = 30
            a3_plot.bgcolor = "white"

            # Attach some tools to the plot
            a3_plot.tools.append(PanTool(a3_plot,
                                         constrain_key="shift"))
            a3_plot.overlays.append(ZoomTool(component=a3_plot,
                                             tool_mode="box", always_on=False))

            self.a3_viewer.plot = a3_plot

    def process_frame(self,buf,buf_offset,timestamp,framenumber):
        """do the work"""

        left_angle_degrees = np.nan
        right_angle_degrees = np.nan


        if self.processing_enabled:
            frame = np.asarray(buf)

            downsampled = np.array(frame[::4,::4])
            bad_cond = (downsampled < self.mask_thresh).astype(np.uint8)
            if True:
                fi_bad_cond = FastImage.asfastimage(bad_cond)
                # use ROI to prevent edge effects
                h,w = bad_cond.shape
                fi_bad_cond_roi = fi_bad_cond.roi(1,1,FastImage.Size(w-2,h-2))
                for i in range(self.n_iterations_dilation):
                    fi_bad_cond_roi_new = fi_bad_cond_roi.dilate3x3(fi_bad_cond_roi.size)
                    fi_bad_cond_roi_new.get_8u_copy_put( fi_bad_cond_roi, fi_bad_cond_roi.size)
                bad_cond = np.asarray(fi_bad_cond)>0
            else:
                if self.n_iterations_dilation:
                    bad_cond = scipy.ndimage.binary_dilation(bad_cond,iterations=self.n_iterations_dilation)

            if self.update_plots:

                self.bg_pd.set_data("imagedata", bad_cond*255)
                self.bg_viewer.plot.request_redraw()
                    #print 'plotting range %f-%f'%(np.min(bad_cond),np.max(bad_cond))

            # blur
            if self.gaussian_sigma != 0.0:
                #blurred = scipy.ndimage.gaussian_filter(f32, self.gaussian_sigma)
                if 1:
                    f32 = downsampled.astype(np.float32)
                    blurred = scipy.ndimage.gaussian_filter1d(f32, self.gaussian_sigma, axis=1)
                    blurred_fi = FastImage.asfastimage(blurred)
                else:
                    fi_downsampled = FastImage.asfastimage(downsampled)
                    blurred_fi_8u = fi_downsampled.gauss5x5(fi_downsampled.size)
                    #blurred_8u = np.asarray(blurred_fi_8u)
                    blurred_fi = blurred_fi_8u.get_32f_copy(blurred_fi_8u.size)
            else:
                f32 = downsampled.astype(np.float32)
                blurred = f32
                blurred_fi = FastImage.asfastimage(blurred)

            G_x = blurred_fi.sobel_horiz(blurred_fi.size)
            G_y = blurred_fi.sobel_vert(blurred_fi.size)

            # square result (abs() would probably work, too)
            G_x.toself_square(G_x.size)         # G_x = G_x**2
            G_y.toself_square(G_y.size)         # G_y = G_y**2
            G_x.toself_add(G_y,G_y.size) # G_x = G_x + G_y
            npG_x = np.asarray(G_x)
            npG_x[bad_cond] = 0 # no edges here -- below self.mask_thresh
            binarize=True
            if binarize:
                #maxG = np.max(npG_x)*0.2
                #maxG = 100.0
                maxG = self.maxG
                binG = (npG_x > maxG).astype(np.uint8)
                if self.update_plots:
                    #print 'np.min(binG),np.max(binG)',np.min(binG),np.max(binG)
                    self.a1_pd.set_data("imagedata", binG*255)
                    self.a1_viewer.plot.request_redraw()
            else:
                if self.update_plots:
                    self.a1_pd.set_data("imagedata", npG_x)
                    self.a1_viewer.plot.request_redraw()
            if 0:
                # no edges here
                npG_x[0,:]=0
                npG_x[-1,:]=0
                npG_x[:,0]=0
                npG_x[:,-1]=0
            #G_x_uint8 = (npG_x).astype(np.uint8)
            #print 'G_x[:5,:5]',npG_x[:5,:5]
            #print 'np.max(G_x)',np.max(npG_x)

            if 1:
                fi_binG = FastImage.asfastimage(binG)
                if binarize:
                    left_mat_quarter = self.strokelitude_instance.maskdata.left_mat_quarter
                    right_mat_quarter = self.strokelitude_instance.maskdata.right_mat_quarter

                    fi_binG = FastImage.asfastimage(binG)
                    left_vals  = compute_sparse_mult(left_mat_quarter,fi_binG)
                    right_vals = compute_sparse_mult(right_mat_quarter,fi_binG)
                    normed_left_vals = left_vals*self.strokelitude_instance.maskdata.left_mat_quarter_norm
                    normed_right_vals = right_vals*self.strokelitude_instance.maskdata.right_mat_quarter_norm
                else:
                    left_mat_half = self.strokelitude_instance.maskdata.left_mat_half_float32
                    right_mat_half = self.strokelitude_instance.maskdata.right_mat_half_float32

                    fi_G_x = FastImage.asfastimage(npG_x)
                    left_vals  = compute_sparse_mult(left_mat_half,fi_G_x)
                    right_vals = compute_sparse_mult(right_mat_half,fi_G_x)
                ## left_vals  = compute_sparse_mult(left_mat_half,fi_G_x_uint8)
                ## right_vals = compute_sparse_mult(right_mat_half,fi_G_x_uint8)

                #print 'left_vals',left_vals
                idx_left = np.argmax(normed_left_vals)
                #print idx_left
                idx_right = np.argmax(normed_right_vals)

                # do we have enough line strength?
                n_pts_left = left_vals[idx_left]
                if n_pts_left >= self.line_strength_thresh:
                    left_angle_radians = self.strokelitude_instance.maskdata.index2angle('left',
                                                                                         idx_left)
                    left_angle_degrees = left_angle_radians*R2D

                n_pts_right = right_vals[idx_right]
                if n_pts_right >= self.line_strength_thresh:
                    right_angle_radians = self.strokelitude_instance.maskdata.index2angle('right',
                                                                                          idx_right)
                    right_angle_degrees = right_angle_radians*R2D
        return left_angle_degrees, right_angle_degrees

class StrokelitudeClass(traited_plugin.HasTraits_FViewPlugin):
    plugin_name = traits.Str('-=| strokelitude |=-')

    # the meta-class, contains startup() and shutdown:
    current_stimulus_plugin = traits.Instance(strokelitude_plugin_module.PluginInfoBase)
    avail_stim_plugins = traits.List(strokelitude_plugin_module.PluginInfoBase)

    # the proxy for the real plugin:
    hastraits_proxy = traits.Instance(traits.HasTraits)
    hastraits_ui = traits.Instance(traits.HasTraits)
    edit_stimulus = traits.Button

    current_amplitude_method = traits.Instance(AmplitudeFinder)
    avail_amplitude_methods = traits.List(AmplitudeFinder)

    draw_mask = traits.Bool

    latency_msec = traits.Float()
    save_to_disk = traits.Bool(False)
    streaming_filename = traits.File
    timestamp_modeler = traits.Instance(
        modeler_module.LiveTimestampModelerWithAnalogInput )

    maskdata = traits.Instance(MaskData)

    # Antenna tracking
    antennae_tracking_enabled = traits.Bool(False)
    write_data_to_a_text_file = traits.Bool(False)
    follow_antennae_enabled = traits.Bool(False)
    draw_antennae_boxes = traits.Bool(False)

    # Slider to define the pixel_threshold
    pixel_threshold = traits.Range(0, 255, 150, mode='slider', set_enter=True)

    # Slider to define the head tracking pivot point
    head_pivot_point_x = traits.Range(0, 656, 335, mode='slider', set_enter=True)
    head_pivot_point_y = traits.Range(0, 656, 245, mode='slider', set_enter=True)


    # Slider to define the width of the box following the antennae
    antenna_box_size = traits.Range(0, 50, 22, mode='slider', set_enter=True)

    # Slider to define the position of the left antenna box
    A_antenna_x = traits.Range(0, 656, 312, mode='slider', set_enter=True)
    A_antenna_y = traits.Range(0, 656, 320, mode='slider', set_enter=True)

    # Slider to define the position of the right antenna box
    B_antenna_x = traits.Range(0, 656, 372, mode='slider', set_enter=True)
    B_antenna_y = traits.Range(0, 656, 320, mode='slider', set_enter=True)

    do_not_save_CamTrig_timestamps = traits.Bool(False)

    # Variables to define the dynamic range based on the frame size
    zero = 0
    max_x = traits.Int
    max_y = traits.Int
    dynamic_min_x = traits.Range(0, 656, 0)
    dynamic_min_y = traits.Range(0, 656, 0)
    dynamic_max_x = traits.Range(0, 656, 656)
    dynamic_max_y = traits.Range(0, 656, 656)

    traits_view = View( Group( Item(name='latency_msec',
                                    label='latency (msec)',
                                    style='readonly',
                                    ),

                               Item('current_amplitude_method',
                                    editor=InstanceEditor(
                                    name = 'avail_amplitude_methods',
                                    selectable=True,
                                    editable = True),
                                    style='custom'),

                           Group(
                               Group( Item('antennae_tracking_enabled'),
                                      Item('follow_antennae_enabled'),
                                      Item('draw_antennae_boxes'),
                                      orientation = 'horizontal'),
                           Group(
                               Group(Item('pixel_threshold', width = 248),
                                     Item('antenna_box_size', width = 248),
                                     orientation = 'vertical',
                                     show_border = True ),

                               Group(Item( 'head_pivot_point_x',
                                            editor = RangeEditor( low_name    = 'zero',
                                                                  high_name   = 'max_x',
                                                                  mode        = 'slider' ),
                                            label = 'X',
                                            width = 350,

                                     ),
                                     Item( 'head_pivot_point_y',
                                            editor = RangeEditor( low_name    = 'zero',
                                                                  high_name   = 'max_y',
                                                                  mode        = 'slider' ),
                                            label = 'Y',
                                            width = 350,

                                     ),
                                     orientation = 'vertical',
                                     label = 'Head pivot point coordinates',
                                     show_border = True ),
                               orientation = 'horizontal'),

                               Group(
                               Group(Item( 'A_antenna_x',
                                            editor = RangeEditor( low_name    = 'dynamic_min_x',
                                                                  high_name   = 'dynamic_max_x',
                                                                  mode        = 'slider' ),
                                            label = 'X',
                                            width = 350,

                                     ),
                                     Item( 'A_antenna_y',
                                            editor = RangeEditor( low_name    = 'dynamic_min_y',
                                                                  high_name   = 'dynamic_max_y',
                                                                  mode        = 'slider' ),
                                            label = 'Y',
                                            width = 350,

                                     ),
                                     orientation = 'vertical',
                                     label = 'Box A coordinates',
                                     show_border = True,
                                     padding = True ),

                               Group(Item( 'B_antenna_x',
                                            editor = RangeEditor( low_name    = 'dynamic_min_x',
                                                                  high_name   = 'dynamic_max_x',
                                                                  mode        = 'slider' ),
                                            label = 'X',
                                            width = 350,

                                     ),
                                     Item( 'B_antenna_y',
                                            editor = RangeEditor( low_name    = 'dynamic_min_y',
                                                                  high_name   = 'dynamic_max_y',
                                                                  mode        = 'slider' ),
                                            label = 'Y',
                                            width = 350,

                                     ),
                                     orientation = 'vertical',
                                     label = 'Box B coordinates',
                                     show_border = True ),
                               orientation = 'horizontal'),

                               label = 'Antennae tracking',
                               show_border = True ),

                               Group(
                               Item('current_stimulus_plugin',
                                    show_label=False,
                                    editor=InstanceEditor(
                                    name = 'avail_stim_plugins',
                                    selectable=True,
                                    editable = False),
                                    style='custom'),
                               Item('edit_stimulus',show_label=False),
                               orientation='horizontal'),

                               Item('draw_mask'),

                               Item('maskdata',show_label=False),

                               Item(name='save_to_disk'),
                               Item('write_data_to_a_text_file'),
                               Item(name='streaming_filename',
                                    style='readonly'),
                               ))

    def __init__(self,*args,**kw):
        kw['wxFrame args']=(-1,self.plugin_name,wx.DefaultPosition,wx.Size(800,750))
        super(StrokelitudeClass,self).__init__(*args,**kw)

        self._list_of_timestamp_data = []
        self._list_of_ain_wordstream_buffers = []

        self.timestamp_modeler = None
        self.streaming_file = None
        self.stream_ain_table   = None
        self.stream_time_data_table = None
        self.stream_table   = None
        self.stream_plugin_tables = None
        self.plugin_table_dtypes = None
        self.current_plugin_descr_dict = {}
        self.display_text_queue = None

        self._list_of_timestamp_data = []
        self._list_of_ain_wordstream_buffers = []

        # Variable used for writing the antenna, wing and head angles to a file
        # See StrokelitudeClass.process_frame method for implementation
        self.first_run = 1

        # Arrays for keeping track of the last couple of eccentricity values
        # Used to avoid interruption from the fly's feet
        self.A_eccentricity_array = np.zeros(30)
        self.B_eccentricity_array = np.zeros(30)

        # load maskdata from file
        self.pkl_fname = motmot.utils.config.rc_fname(
            must_already_exist=False,
            filename='strokelitude-maskdata.pkl',
            dirname='.fview')
        loaded_maskdata = False
        if os.path.exists(self.pkl_fname):
            try:
                self.maskdata = pickle.load(open(self.pkl_fname))
                loaded_maskdata = True
            except Exception,err:
                warnings.warn(
                    'could not open strokelitude persistance file: %s'%err)
        if not loaded_maskdata:
            self.maskdata = MaskData()
        self.save_data_queue = Queue.Queue()

        # for passing data to the plugin
        self.current_plugin_queue = None
        # for saving data:
        self.current_plugin_save_queues = {} # replaced later

        if 1:
            # load analysis methods
            self.avail_amplitude_methods.append(SobelFinder())
            self.avail_amplitude_methods.append(BackgroundSubtractionDotProductFinder())
            self.current_amplitude_method = self.avail_amplitude_methods[0]
            for method in self.avail_amplitude_methods:
                method.post_init(self)

        if 1:
            # load plugins
            plugins = load_plugins()
            for p in plugins:
                self.avail_stim_plugins.append(p)
            if len(self.avail_stim_plugins):
                self.current_stimulus_plugin = self.avail_stim_plugins[0]
            self.quit_plugin_event = threading.Event()

        self.cam_id = None
        self.width = 20
        self.height = 10

        ID_Timer = wx.NewId()
        self.timer = wx.Timer(self.frame, ID_Timer)
        wx.EVT_TIMER(self.frame, ID_Timer, self.OnTimer)
        self.timer.Start(2000)

        if have_ROS:
            rospy.init_node('fview', # common name across all plugins so multiple calls to init_node() don't fail
                            anonymous=True, # allow multiple instances to run
                            disable_signals=True, # let WX intercept them
                            )
            self.publisher_lock = threading.Lock()
            self.publisher = rospy.Publisher('strokelitude',
                                             FlyWingEstimate,
                                             tcp_nodelay=True,
                                             )

    def _timestamp_modeler_changed(self,newvalue):
        # register our handlers
        self.timestamp_modeler.on_trait_change(self.on_ain_data_raw,
                                               'ain_data_raw')
        self.timestamp_modeler.on_trait_change(self.on_timestamp_data,
                                               'timestamps_framestamps')
    def on_ain_data_raw(self,newvalue):
        self._list_of_ain_wordstream_buffers.append(newvalue)

    def on_timestamp_data(self,timestamp_framestamp_2d_array):
        if len(timestamp_framestamp_2d_array):
            last_sample = timestamp_framestamp_2d_array[-1,:]
            self._list_of_timestamp_data.append(last_sample)

    def OnTimer(self,event):
        self.service_save_data()
        if self.display_text_queue is not None:
            while self.display_text_queue.qsize() > 0:
                print('self.display_text_queue.get_nowait()',
                      self.display_text_queue.get_nowait())

    def on_ain_data_raw(self,newvalue):
        self._list_of_ain_wordstream_buffers.append(newvalue)

    def on_timestamp_data(self,timestamp_framestamp_2d_array):
        if len(timestamp_framestamp_2d_array):
            last_sample = timestamp_framestamp_2d_array[-1,:]
            self._list_of_timestamp_data.append(last_sample)

    def service_save_data(self):
        # pump the queue
        list_of_rows_of_data = []
        try:
            while 1:
                list_of_rows_of_data.append( self.save_data_queue.get_nowait() )
        except Queue.Empty:
            pass

        if self.stream_table is not None and len(list_of_rows_of_data):
            # it's much faster to convert to numpy first:
            recarray = np.rec.array(list_of_rows_of_data,
                                    dtype=StrokelitudeDataCol_dtype)
            self.stream_table.append( recarray )
            self.stream_table.flush()

        # analog input data...
        bufs = self._list_of_ain_wordstream_buffers
        self._list_of_ain_wordstream_buffers = []
        if self.stream_ain_table is not None and len(bufs):
            buf = np.hstack(bufs)
            recarray = np.rec.array( [buf], dtype=AnalogInputWordstream_dtype)
            self.stream_ain_table.append( recarray )
            self.stream_ain_table.flush()

        tsfss = self._list_of_timestamp_data
        self._list_of_timestamp_data = []
        if self.stream_time_data_table is not None and len(tsfss):
            bigarr = np.vstack(tsfss)
            timestamps = bigarr[:,0]
            framestamps = bigarr[:,1]
            recarray = np.rec.array( [timestamps,framestamps], dtype=TimeData_dtype)
            self.stream_time_data_table.append( recarray )
            self.stream_time_data_table.flush()

        # plugin data...
        for name,queue in self.current_plugin_save_queues.iteritems():

            list_of_rows_of_data = []
            try:
                while 1:
                    list_of_rows_of_data.append( queue.get_nowait() )
            except Queue.Empty:
                pass

            if self.stream_plugin_tables is not None and len(list_of_rows_of_data):
                recarray = np.rec.array(list_of_rows_of_data,
                                        dtype=self.plugin_table_dtypes[name])
                table = self.stream_plugin_tables[name]
                table.append(recarray)
                table.flush()

    def _save_to_disk_changed(self):
        self.service_save_data()
        if self.save_to_disk:
            if self.timestamp_modeler is not None:
                self.timestamp_modeler.block_activity = True

            self.streaming_filename = time.strftime('strokelitude%Y%m%d_%H%M%S.h5')
            self.streaming_file = tables.openFile( self.streaming_filename, mode='w')
            self.stream_ain_table   = self.streaming_file.createTable(
                self.streaming_file.root,'ain_wordstream',AnalogInputWordstreamDescription,
                "AIN data",expectedrows=100000)
            if self.timestamp_modeler is not None:
                names = self.timestamp_modeler.channel_names
                print 'saving analog channels',names
                self.stream_ain_table.attrs.channel_names = names

                self.stream_ain_table.attrs.Vcc = self.timestamp_modeler.Vcc

            self.stream_time_data_table = self.streaming_file.createTable(
                self.streaming_file.root,'time_data',TimeDataDescription,
                "time data",expectedrows=10000)
            if self.timestamp_modeler is not None:
                self.stream_time_data_table.attrs.top = self.timestamp_modeler.timer3_top

            self.stream_table   = self.streaming_file.createTable(
                self.streaming_file.root,'stroke_data', StrokelitudeDataDescription,
                "wingstroke data",expectedrows=50*60*10) # 50 Hz * 60 seconds * 10 minutes

            self.stream_plugin_tables = {}
            self.plugin_table_dtypes = {}
            for name, description in self.current_plugin_descr_dict.iteritems():
                self.stream_plugin_tables[name] = self.streaming_file.createTable(
                    self.streaming_file.root, name, description,
                    name,expectedrows=50*60*10) # 50 Hz * 60 seconds * 10 minutes
                self.plugin_table_dtypes[name] = tables.Description(
                    description().columns)._v_nestedDescr

            print 'saving to disk...', os.path.abspath(self.streaming_filename)
        else:
            print 'closing file...'
            # clear queue
            self.save_data_queue = Queue.Queue()

            self.stream_ain_table   = None
            self.stream_time_data_table = None
            self.stream_table   = None
            self.stream_plugin_tables = None
            self.plugin_table_dtypes = None
            #if self.streaming_file is not None:
            self.streaming_file.close()
            self.streaming_file = None
            print 'closed',repr(self.streaming_filename)
            self.streaming_filename = ''
            if self.timestamp_modeler is not None:
                self.timestamp_modeler.block_activity = False

    def _edit_stimulus_fired(self):
        if self.current_stimulus_plugin is not None:
            self.hastraits_ui = self.hastraits_proxy.edit_traits()

    # handler for change of self.current_stimulus_plugin
    def _current_stimulus_plugin_changed(self,trait_name,old_value,new_value):
        print 'trait_name,old_value,new_value',trait_name,old_value,new_value

        print 'calling _current_stimulus_plugin_changed...'
        if self.save_to_disk:
            print 'ERROR: chose plugin while saving to disk'

        if self.hastraits_ui is not None:
            print 'do not know how to close old window!!'
            #self.hastraits_ui.close()

        if old_value is not None:
            # shutdown old plugin
            old_value.shutdown()
            self.current_plugin_queue = None

        assert new_value is self.current_stimulus_plugin

        # startup new plugin

        print 'calling startup()'
        (self.hastraits_proxy, self.current_plugin_queue,
         self.current_plugin_save_queues,
         self.current_plugin_descr_dict,
         self.display_text_queue) = self.current_stimulus_plugin.startup()
        self.hastraits_ui = None

        print '...done _current_stimulus_plugin_changed'

    def get_buffer_allocator(self,cam_id):
        return BufferAllocator()

    def process_frame(self,cam_id,buf,buf_offset,timestamp,framenumber):
        """do work on each frame

        This function gets called on every single frame capture. It is
        called within the realtime thread, NOT the wxPython
        application mainloop's thread. Therefore, be extremely careful
        (use threading locks) when sharing data with the rest of the
        class.

        """
        draw_points = [] #  [ (x,y) ]
        draw_linesegs = [] # [ (x0,y0,x1,y1) ]

        x_offset = buf_offset[0]
        y_offset = buf_offset[1]

        if not self.do_not_save_CamTrig_timestamps:
            if self.timestamp_modeler is not None:
                trigger_timestamp = self.timestamp_modeler.register_frame(
                    cam_id,framenumber,timestamp)
            else:
                trigger_timestamp = None
        else:
            # take trigger timestamp from fmf file on replay
            trigger_timestamp = timestamp

        tmp = self.current_amplitude_method.process_frame(buf,buf_offset,timestamp,framenumber)
        left_angle_degrees, right_angle_degrees = tmp

        if self.current_amplitude_method.processing_enabled and have_ROS:
            msg = FlyWingEstimate()
            msg.header.seq=framenumber
            if timestamp is not None:
                msg.header.stamp=rospy.Time.from_sec(timestamp)
            else:
                msg.header.stamp=rospy.Time.from_sec(0.0)
            msg.header.frame_id = "0"
            msg.left_wing_angle_radians = left_angle_degrees * D2R
            msg.right_wing_angle_radians = right_angle_degrees * D2R
            if have_ROS:
                with self.publisher_lock:
                    self.publisher.publish(msg)

        # draw lines
        for side, angle_degrees in [('left',left_angle_degrees),
                                    ('right',right_angle_degrees)]:
            if not np.isnan(angle_degrees):
                this_seg = self.maskdata.get_span_lineseg(side,angle_degrees*D2R)
                draw_linesegs.extend(this_seg)

        processing_timestamp = time.time()
        if trigger_timestamp is not None:
            self.latency_msec = (processing_timestamp-trigger_timestamp)*1000.0

        if self.current_plugin_queue is not None:
            self.current_plugin_queue.put(
                (framenumber,left_angle_degrees, right_angle_degrees,
                trigger_timestamp) )

        if self.draw_mask:
            # XXX this is naughty -- it's not threasafe.
            # concatenate lists
            extra = self.maskdata.all_linesegs

            draw_linesegs.extend( extra )

        if self.antennae_tracking_enabled:
            tmp = self
            frame = np.asarray(buf)

            # Parameters for drawing the antennae boxes
            big = math.ceil(tmp.antenna_box_size*0.7)
            small = math.ceil(tmp.antenna_box_size*0.3)

            self.max_x = frame.shape[1]
            self.max_y = frame.shape[0]
            self.dynamic_min_x = int(big)
            self.dynamic_max_x = int(math.floor(self.max_x - big))
            self.dynamic_min_y = int(math.ceil(small))
            self.dynamic_max_y = int(math.floor(self.max_y - big))

            border_violation = tmp.A_antenna_x < self.dynamic_min_x or \
                    tmp.A_antenna_x > self.dynamic_max_x or \
                    tmp.A_antenna_y < self.dynamic_min_y or \
                    tmp.A_antenna_y > self.dynamic_max_y or \
                    tmp.B_antenna_x < self.dynamic_min_x or \
                    tmp.B_antenna_x > self.dynamic_max_x or \
                    tmp.B_antenna_y < self.dynamic_min_y or \
                    tmp.B_antenna_y > self.dynamic_max_y

            if border_violation:
                pass

            else:
                tmp.A_antenna_x0 = tmp.A_antenna_x - big
                tmp.A_antenna_x1 = tmp.A_antenna_x + small
                tmp.A_antenna_y0 = tmp.A_antenna_y - small
                tmp.A_antenna_y1 = tmp.A_antenna_y + big
                tmp.B_antenna_x0 = tmp.B_antenna_x - small
                tmp.B_antenna_x1 = tmp.B_antenna_x + big
                tmp.B_antenna_y0 = tmp.B_antenna_y - small
                tmp.B_antenna_y1 = tmp.B_antenna_y + big
                tmp.A_antenna_x0_offset = tmp.A_antenna_x - big + x_offset
                tmp.A_antenna_x1_offset = tmp.A_antenna_x + small + x_offset
                tmp.A_antenna_y0_offset = tmp.A_antenna_y - small + y_offset
                tmp.A_antenna_y1_offset = tmp.A_antenna_y + big + y_offset
                tmp.B_antenna_x0_offset = tmp.B_antenna_x - small + x_offset
                tmp.B_antenna_x1_offset = tmp.B_antenna_x + big + x_offset
                tmp.B_antenna_y0_offset = tmp.B_antenna_y - small + y_offset
                tmp.B_antenna_y1_offset = tmp.B_antenna_y + big + y_offset

            if 1:
                # Cropped regions of the antennae
                region_A = (255 - frame[tmp.A_antenna_y0:tmp.A_antenna_y1,
                              tmp.A_antenna_x0:tmp.A_antenna_x1]) > (255 - tmp.pixel_threshold)
                region_B = (255 - frame[tmp.B_antenna_y0:tmp.B_antenna_y1,
                              tmp.B_antenna_x0:tmp.B_antenna_x1]) > (255 - tmp.pixel_threshold)

                fi_region_A = FastImage.asfastimage(region_A.astype(np.uint8))
                fi_region_B = FastImage.asfastimage(region_B.astype(np.uint8))


                # Error handling when the fly's leg gets inside the antennae box
                try:
                    A_fitting = realtime_image_analysis.FitParamsClass().fit(fi_region_A)
                    B_fitting = realtime_image_analysis.FitParamsClass().fit(fi_region_B)

                    # Parameters for drawing lines along the antennae
                    A_area = A_fitting[2]
                    A_x0 = A_fitting[0]
                    A_y0 = A_fitting[1]
                    A_slope = A_fitting[3]
                    A_eccentricity = A_fitting[4]
                    self.A_eccentricity_array = np.append(self.A_eccentricity_array,A_eccentricity)
                    self.A_eccentricity_array = self.A_eccentricity_array.__getslice__(1,self.A_eccentricity_array.__len__())

                    B_area = B_fitting[2]
                    B_x0 = B_fitting[0]
                    B_y0 = B_fitting[1]
                    B_slope = B_fitting[3]
                    B_eccentricity = B_fitting[4]
                    self.B_eccentricity_array = np.append(self.B_eccentricity_array,B_eccentricity)
                    self.B_eccentricity_array = self.B_eccentricity_array.__getslice__(1,self.B_eccentricity_array.__len__())

                    # Length of the lines drawn along the antennae
                    r = 40

                    if tmp.maskdata.view_from_below:

                        left_eccentricity_mean = self.A_eccentricity_array.mean()
                        left_x00 = tmp.A_antenna_x0 + A_x0
                        left_y00 = tmp.A_antenna_y0 + A_y0
                        left_theta = np.arctan(A_slope) + math.pi                       # Add pi to get the correct angle
                        left_theta_degrees = (math.pi - left_theta)*180/math.pi         # Angle in degrees with respect to a horizontal line
                        left_dx = r*np.cos(left_theta)
                        left_dy = r*np.sin(left_theta)
                        left_antennae = (left_x00 + x_offset,left_y00 + y_offset,
					 left_x00 + left_dx + x_offset,
					 left_y00 + left_dy + y_offset)

                        right_eccentricity_mean = self.B_eccentricity_array.mean()
                        right_x00 = tmp.B_antenna_x0 + B_x0
                        right_y00 = tmp.B_antenna_y0 + B_y0
                        right_theta = np.arctan(B_slope)
                        right_theta_degrees = (right_theta)*180/math.pi                 # Angle in degrees from a vertical line along the fly
                        right_dx = r*np.cos(right_theta)
                        right_dy = r*np.sin(right_theta)
                        right_antennae = (right_x00 + x_offset,right_y00 + y_offset,
					  right_x00 + right_dx + x_offset,
					  right_y00 + right_dy + y_offset)

                    else:

                        left_eccentricity_mean = self.B_eccentricity_array.mean()
                        left_x00 = tmp.B_antenna_x0 + B_x0
                        left_y00 = tmp.B_antenna_y0 + B_y0
                        left_theta = np.arctan(B_slope)
                        left_theta_degrees = (left_theta)*180/math.pi                   # Angle in degrees with respect to a horizontal line
                        left_dx = r*np.cos(left_theta)
                        left_dy = r*np.sin(left_theta)
                        left_antennae = (left_x00 + x_offset,left_y00 + y_offset,
					 left_x00 + left_dx + x_offset,
					 left_y00 + left_dy + y_offset)

                        right_eccentricity_mean = self.A_eccentricity_array.mean()
                        right_x00 = tmp.A_antenna_x0 + A_x0
                        right_y00 = tmp.A_antenna_y0 + A_y0
                        right_theta = np.arctan(A_slope) + math.pi                      # Add pi to get the correct angle
                        right_theta_degrees = (math.pi - right_theta)*180/math.pi       # Angle in degrees from a vertical line along the fly
                        right_dx = r*np.cos(right_theta)
                        right_dy = r*np.sin(right_theta)
                        right_antennae = (right_x00 + x_offset,right_y00 + y_offset,
					  right_x00 + right_dx + x_offset,
					  right_y00 + right_dy + y_offset)

                    try:
                        left_x00
                        x00_is_set = 1
                    except NameError:
                        x00_is_set = 0

                    if tmp.follow_antennae_enabled and x00_is_set: # and left_eccentricity > 0.8*left_eccentricity_mean and right_eccentricity > 0.8*right_eccentricity_mean:
                        if tmp.maskdata.view_from_below:
                            tmp.A_antenna_x = int(left_x00)
                            tmp.A_antenna_y = int(left_y00)
                            tmp.B_antenna_x = int(right_x00)
                            tmp.B_antenna_y = int(right_y00)
                        else:
                            tmp.A_antenna_x = int(right_x00)
                            tmp.A_antenna_y = int(right_y00)
                            tmp.B_antenna_x = int(left_x00)
                            tmp.B_antenna_y = int(left_y00)

                except realtime_image_analysis.FitParamsError:
                    print 'FitParamsError, antennae angles = NaN.'
                    right_theta_degrees = np.nan
                    left_theta_degrees = np.nan
                    left_antennae = []
                    right_antennae = []
                    x00_is_set = 0
                    letter_L1 = []
                    letter_L2 = []

                if x00_is_set:
                    between_antenna_x = (left_x00 + right_x00)/2
                    between_antenna_y = (left_y00 + right_y00)/2
                    head_slope = (between_antenna_x - tmp.head_pivot_point_x) / (between_antenna_y - tmp.head_pivot_point_y)
                    head_theta = np.arctan(head_slope)
                    if tmp.maskdata.view_from_below:
                        head_theta_degrees = (head_theta)*180/math.pi
                    else:
                        head_theta_degrees = -(head_theta)*180/math.pi

                else:
                    between_antenna_x = np.nan
                    between_antenna_y = np.nan
                    head_theta_degrees = np.nan

            # All angles are interpreted from a horizontal line outward from the fly.
            # A positive head angle means that the head is tilted to the right.
            if tmp.write_data_to_a_text_file:
                # Create and open a file for writing the antennae and head angles only on first run
                # The variable first_run is defined in the beginning of the class StrokelitudeClass
                if self.first_run:
                    os.environ['TZ'] = 'UTC+07'
                    time.tzset()
                    experiment_time = time.strftime("%H%M%S")
                    experiment_date = time.strftime("%Y%m%d")
                    antennae_filename = 'angles_' + experiment_date + '_' + experiment_time + '.txt'
                    self.antennae_angle = open(antennae_filename,'w')
                    self.first_run = 0
                timestamp = repr(time.time()) + '\t'
                antennae_line = repr(left_theta_degrees) + '\t' + repr(right_theta_degrees) + '\t'
                wing_line = repr(left_angle_degrees)+ '\t' + repr(right_angle_degrees) + '\t'
                head_line = repr(head_theta_degrees) + '\n'
                line = timestamp + antennae_line + wing_line + head_line
                self.antennae_angle.write(line)

            self.save_data_queue.put(
                (framenumber,trigger_timestamp,processing_timestamp,
                left_angle_degrees, right_angle_degrees,
                left_theta_degrees, right_theta_degrees, head_theta_degrees))

            if 1:
                # Draw a lines along the antennae
                if len(left_antennae):
                    draw_linesegs.append( left_antennae )
                if len(right_antennae):
                    draw_linesegs.append( right_antennae )

                draw_points.append([tmp.head_pivot_point_x + x_offset,
				    tmp.head_pivot_point_y + y_offset])

                draw_linesegs.append( (tmp.head_pivot_point_x + x_offset,
				       tmp.head_pivot_point_y + y_offset,
                                       between_antenna_x + x_offset,
				       between_antenna_y + y_offset) )

            if tmp.draw_antennae_boxes:
                # Draw a box around the left antenna
                draw_linesegs.append( (tmp.A_antenna_x0_offset, tmp.A_antenna_y0_offset,
                                       tmp.A_antenna_x0_offset, tmp.A_antenna_y1_offset) )
                draw_linesegs.append( (tmp.A_antenna_x0_offset, tmp.A_antenna_y1_offset,
                                       tmp.A_antenna_x1_offset, tmp.A_antenna_y1_offset) )
                draw_linesegs.append( (tmp.A_antenna_x1_offset, tmp.A_antenna_y1_offset,
                                       tmp.A_antenna_x1_offset, tmp.A_antenna_y0_offset) )
                draw_linesegs.append( (tmp.A_antenna_x1_offset, tmp.A_antenna_y0_offset,
                                       tmp.A_antenna_x0_offset, tmp.A_antenna_y0_offset) )

                                # Draw a box around the right antenna
                draw_linesegs.append( (tmp.B_antenna_x0_offset, tmp.B_antenna_y0_offset,
                                       tmp.B_antenna_x0_offset, tmp.B_antenna_y1_offset) )
                draw_linesegs.append( (tmp.B_antenna_x0_offset, tmp.B_antenna_y1_offset,
                                       tmp.B_antenna_x1_offset, tmp.B_antenna_y1_offset) )
                draw_linesegs.append( (tmp.B_antenna_x1_offset, tmp.B_antenna_y1_offset,
                                       tmp.B_antenna_x1_offset, tmp.B_antenna_y0_offset) )
                draw_linesegs.append( (tmp.B_antenna_x1_offset, tmp.B_antenna_y0_offset,
                                       tmp.B_antenna_x0_offset, tmp.B_antenna_y0_offset) )

                # Draw a letter indicating the left side
                if tmp.maskdata.view_from_below:
                    letter_L1 = [tmp.A_antenna_x1 + x_offset, tmp.A_antenna_y1 + y_offset + 5,
				 tmp.A_antenna_x1 + x_offset, tmp.A_antenna_y1 + y_offset + 20]
                    letter_L2 = [tmp.A_antenna_x1 + x_offset, tmp.A_antenna_y1 + y_offset + 5,
				 tmp.A_antenna_x1 + x_offset - 10, tmp.A_antenna_y1 + y_offset + 5]
                else:
                    letter_L1 = [tmp.B_antenna_x1 + x_offset, tmp.B_antenna_y1 + y_offset + 5,
				 tmp.B_antenna_x1 + x_offset, tmp.B_antenna_y1 + y_offset + 20]
                    letter_L2 = [tmp.B_antenna_x1 + x_offset, tmp.B_antenna_y1 + y_offset + 5,
				 tmp.B_antenna_x1 + x_offset - 10, tmp.B_antenna_y1 + y_offset + 5]

                draw_linesegs.append( letter_L1 )
                draw_linesegs.append( letter_L2 )

        return draw_points, draw_linesegs

    def camera_starting_notification(self,cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):
        if self.cam_id is not None:
            raise NotImplementedError('only a single camera is supported.')
        self.cam_id = cam_id

        self.width = max_width
        self.height = max_height
        self.maskdata.maxx = self.width
        self.maskdata.maxy = self.height

        for method in self.avail_amplitude_methods:
            method.camera_starting_notification(cam_id,
                                                pixel_format=pixel_format,
                                                max_width=max_width,
                                                max_height=max_height)

    def offline_startup_func(self,arg):
        """gets called by fview_replay_fmf"""
        self.do_not_save_CamTrig_timestamps = True
        for method in self.avail_amplitude_methods:
            method.offline_startup_func(arg)

    def set_all_fview_plugins(self,plugins):
        print 'set_all_fview_plugins called'
        for plugin in plugins:
            print '  ', plugin.get_plugin_name()

            if plugin.get_plugin_name()=='FView external trigger':
                self.timestamp_modeler = plugin.timestamp_modeler
        print

    def _timestamp_modeler_changed(self,newvalue):
        # register our handlers
        self.timestamp_modeler.on_trait_change(self.on_ain_data_raw,
                                               'ain_data_raw')
        self.timestamp_modeler.on_trait_change(self.on_timestamp_data,
                                               'timestamps_framestamps')

    def quit(self):
        # save current maskdata
        pickle.dump(self.maskdata,open(self.pkl_fname,mode='w'))

        if self.current_stimulus_plugin is not None:
            self.current_stimulus_plugin.shutdown()

if __name__=='__main__':
    data = MaskData()
    data.configure_traits()
