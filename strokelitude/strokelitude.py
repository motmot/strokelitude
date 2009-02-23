from __future__ import division, with_statement

import pkg_resources
import data_format # strokelitude

import motmot.utils.config
import motmot.FastImage.FastImage as FastImage
import motmot.fview.traited_plugin as traited_plugin

# Weird bugs arise in class identity testing if this is not absolute
#   import...  ... but Python won't let me import it as absolute. Ahh,
#   well, workaround the bug by setting selectable on InstanceEditor
#   directly.
import plugin as strokelitude_plugin_module

import wx
import warnings, time, threading

import numpy as np
import cairo
import Queue
import os, warnings, pickle

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

    mean_quad_angles_deg = traits.Property(depends_on=[
        'alpha', 'beta', 'nbins'])

    quads_right = traits.Property(depends_on=[
        'wingsplit', 'r1', 'r2', 'alpha', 'beta', 'nbins',
        'rotation','translation','view_from_below',
        ])

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

def quad2fastimage_offset(quad,width,height,debug_count=0):
    """convert a quad to an image vector"""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                 width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0,0,0)

    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.paint()

    x0=quad[0]
    y0=quad[1]
    ctx.move_to(x0,y0)
    xmin = int(np.floor(x0))
    xmax = int(np.ceil(x0+1))
    ymin = int(np.floor(y0))
    ymax = int(np.ceil(y0+1))

    for (x,y) in zip(quad[2::2],
                     quad[3::2]):
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

class StrokelitudeClass(traited_plugin.HasTraits_FViewPlugin):
    plugin_name = traits.Str('-=| strokelitude |=-')

    # the meta-class, contains startup() and shutdown:
    current_stimulus_plugin = traits.Instance(strokelitude_plugin_module.PluginInfoBase)
    avail_stim_plugins = traits.List(strokelitude_plugin_module.PluginInfoBase)

    # the proxy for the real plugin:
    hastraits_proxy = traits.Instance(traits.HasTraits)
    hastraits_ui = traits.Instance(traits.HasTraits)
    edit_stimulus = traits.Button

    mask_dirty = traits.Bool(False) # let __init__ set True
    maskdata = traits.Instance(MaskData)

    recompute_mask = traits.Event
    take_bg = traits.Event
    clear_bg = traits.Event

    background_viewer = traits.Instance(PopUpPlot)
    live_data_viewer = traits.Instance(LiveDataPlot)

    processing_enabled = traits.Bool(False)

    draw_mask = traits.Bool

    latency_msec = traits.Float()
    threshold_fraction = traits.Float(0.5)
    light_on_dark = traits.Bool(True)
    save_to_disk = traits.Bool(False)
    streaming_filename = traits.File
    timestamp_modeler = traits.Instance(
        modeler_module.LiveTimestampModelerWithAnalogInput )

    live_data_plot = traits.Instance(LiveDataPlot)

    traits_view = View( Group( Item(name='latency_msec',
                                    label='latency (msec)',
                                    style='readonly',
                                    ),
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
                               Item('recompute_mask',
                                    editor=ButtonEditor(),show_label=False),
                               Item('processing_enabled'),
                               Item('draw_mask'),
                               Item('take_bg',
                                    editor=ButtonEditor(),show_label=False),
                               Item('clear_bg',
                                    editor=ButtonEditor(),show_label=False),
                               Item('background_viewer',show_label=False),
                               Item('live_data_viewer',show_label=False),
                               Item('maskdata',show_label=False),
                               Item('live_data_plot',show_label=False),
                               Item(name='threshold_fraction',
                                    ),
                               Item(name='light_on_dark',
                                    ),
                               Item(name='save_to_disk',
                                    ),
                               Item(name='streaming_filename',
                                    style='readonly'),
                               ))

    def __init__(self,*args,**kw):
        print 'init called',args,kw
        kw['wxFrame args']=(-1,self.plugin_name,wx.DefaultPosition,wx.Size(800,600))
        super(StrokelitudeClass,self).__init__(*args,**kw)

        self.timestamp_modeler = None
        self.drawsegs_cache = None
        self.streaming_file = None
        self.stream_ain_table   = None
        self.stream_time_data_table = None
        self.stream_table   = None
        self.stream_plugin_tables = None
        self.plugin_table_dtypes = None
        self.current_plugin_descr_dict = {}
        self.display_text_queue = None

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
        self.maskdata.on_trait_change( self.on_mask_change )
        self.save_data_queue = Queue.Queue()

        self.vals_queue = Queue.Queue()
        self.bg_cmd_queue = Queue.Queue()
        self.new_bg_image_lock = threading.Lock()
        self.new_bg_image = None

        self.recomputing_lock = threading.Lock()
        # for passing data to the pluging
        self.current_plugin_queue = None
        # for saving data:
        self.current_plugin_save_queues = {} # replaced later

        self.background_viewer = PopUpPlot()
        self.live_data_plot = LiveDataPlot()

        if 1:
            # load plugins
            plugins = load_plugins()
            for p in plugins:
                self.avail_stim_plugins.append(p)
            if len(self.avail_stim_plugins):
                self.current_stimulus_plugin = self.avail_stim_plugins[0]
            self.quit_plugin_event = threading.Event()

        if 1:
            self.live_pd = {}
            for side in ['left','right']:

                x=self.maskdata.mean_quad_angles_deg
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
        self.cam_id = None
        self.width = 20
        self.height = 10

        self.frame.Connect( -1, -1, DataReadyEvent, self.OnDataReady )
        self.frame.Connect( -1, -1, BGReadyEvent, self.OnNewBGReady )
        self.mask_dirty = True

        ID_Timer = wx.NewId()
        self.timer = wx.Timer(self.frame, ID_Timer)
        wx.EVT_TIMER(self.frame, ID_Timer, self.OnTimer)
        self.timer.Start(2000)

    def OnTimer(self,event):
        self.service_save_data()
        if self.display_text_queue is not None:
            while self.display_text_queue.qsize() > 0:
                print('self.display_text_queue.get_nowait()',
                      self.display_text_queue.get_nowait())

    def service_save_data(self,flush=False):
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
        buf = self.timestamp_modeler.pump_ain_wordstream_buffer(flush=flush)
        if self.stream_ain_table is not None and buf is not None:
            recarray = np.rec.array( [buf], dtype=AnalogInputWordstream_dtype)
            self.stream_ain_table.append( recarray )
            self.stream_ain_table.flush()

        tsfs = self.timestamp_modeler.pump_timestamp_data(flush=flush)
        if self.stream_time_data_table is not None and tsfs is not None:
            timestamps,framestamps = tsfs
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

    def _mask_dirty_changed(self):
        if self.mask_dirty:
            self.processing_enabled = False

    def _save_to_disk_changed(self):
        self.service_save_data(flush=True) # flush buffers
        if self.save_to_disk:
            self.timestamp_modeler.block_activity = True

            self.streaming_filename = time.strftime('strokelitude%Y%m%d_%H%M%S.h5')
            self.streaming_file = tables.openFile( self.streaming_filename, mode='w')
            self.stream_ain_table   = self.streaming_file.createTable(
                self.streaming_file.root,'ain_wordstream',AnalogInputWordstreamDescription,
                "AIN data",expectedrows=100000)
            self.stream_ain_table.attrs.channel_names = self.timestamp_modeler.channel_names

            self.stream_time_data_table = self.streaming_file.createTable(
                self.streaming_file.root,'time_data',TimeDataDescription,
                "time data",expectedrows=10000)
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

            print 'saving to disk...'
        else:
            print 'closing file...'
            # flush queue
            self.save_data_queue = Queue.Queue()

            self.stream_ain_table   = None
            self.stream_time_data_table = None
            self.stream_table   = None
            self.stream_plugin_tables = None
            self.plugin_table_dtypes = None
            self.streaming_file.close()
            self.streaming_file = None
            print 'closed',repr(self.streaming_filename)
            self.streaming_filename = ''
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

    def _recompute_mask_fired(self):
        print 'recomputing mask'
        with self.recomputing_lock:
            count = 0

            left_quads = self.maskdata.quads_left
            right_quads = self.maskdata.quads_right

            self.left_mat = []
            for quad in left_quads:
                fi_roi, left, bottom = quad2fastimage_offset(
                    quad,
                    self.width,self.height,
                    debug_count=count)
                self.left_mat.append( (fi_roi, left, bottom) )
                count+=1

            self.right_mat = []
            for quad in right_quads:
                fi_roi, left, bottom = quad2fastimage_offset(
                    quad,
                    self.width,self.height,
                    debug_count=count)
                self.right_mat.append( (fi_roi, left, bottom) )
                count+=1

            bg = FastImage.asfastimage(self.bg_image)
            print 'self.bg_image[:10,:10]'
            print self.bg_image[:10,:10]
            self.bg_left_vec = compute_sparse_mult(self.left_mat,bg)
            self.bg_right_vec= compute_sparse_mult(self.right_mat,bg)

            self.live_pd['left'].set_data('bg',self.bg_left_vec)
            self.live_pd['right'].set_data('bg',self.bg_right_vec)
            #self.left_plot.request_redraw()
            #self.right_plot.request_redraw()

            print self.bg_left_vec

            self.mask_dirty=False

    def on_mask_change(self):
        self.mask_dirty=True

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

        if self.timestamp_modeler is not None:
            trigger_timestamp = self.timestamp_modeler.register_frame(
                cam_id,framenumber,timestamp)
        else:
            trigger_timestamp = None

        have_lock = self.recomputing_lock.acquire(False) # don't block
        if not have_lock:
            return draw_points, draw_linesegs

        try:
            if self.draw_mask:
                # XXX this is naughty -- it's not threasafe.
                # concatenate lists
                extra = self.maskdata.all_linesegs

                draw_linesegs.extend( extra )

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
                event.SetEventObject(self.frame)

                # trigger call to self.OnNewBGReady
                wx.PostEvent(self.frame, event)

            # XXX naughty to cross thread boundary to get enabled_box value, too
            if (self.processing_enabled and not self.mask_dirty):
                if 1:
                    self.drawsegs_cache = []

                    h,w = this_image.shape
                    if not (self.width==w and self.height==h):
                        raise NotImplementedError('need to support ROI')

                    else:
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
                            angle_radians = self.maskdata.index2angle(side,
                                                                      interp_idx)
                            results.append( angle_radians*R2D ) # keep results in degrees

                            # draw lines
                            this_seg = self.maskdata.get_span_lineseg(side,angle_radians)
                            draw_linesegs.extend(this_seg)
                            self.drawsegs_cache.extend( this_seg )
                        else:
                            results.append( np.nan )

                    left_angle_degrees, right_angle_degrees = results
                    processing_timestamp = time.time()
                    if trigger_timestamp is not None:
                        self.latency_msec = (processing_timestamp-trigger_timestamp)*1000.0

                    if self.current_plugin_queue is not None:
                        self.current_plugin_queue.put(
                            (framenumber,left_angle_degrees, right_angle_degrees,
                            trigger_timestamp) )
                    self.save_data_queue.put(
                        (framenumber,trigger_timestamp,processing_timestamp,
                        left_angle_degrees, right_angle_degrees))

                    ## for queue in self.plugin_data_queues:
                    ##     queue.put( (cam_id,timestamp,framenumber,results) )

                    # send values from each quad to be drawn
                    self.vals_queue.put((left_vals, right_vals,
                                         left_angle_degrees,right_angle_degrees,
                                         ))
                    event = wx.CommandEvent(DataReadyEvent)
                    event.SetEventObject(self.frame)

                    # trigger call to self.OnDataReady
                    wx.PostEvent(self.frame, event)
                else:
                    if self.drawsegs_cache is not None:
                        draw_linesegs.extend( self.drawsegs_cache )

            else:
                self.drawsegs_cache = None
                if trigger_timestamp is not None:
                    now = time.time()
                    self.latency_msec = (now-trigger_timestamp)*1000.0

        finally:
            self.recomputing_lock.release()

        return draw_points, draw_linesegs

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
        if self.save_to_disk:
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

        with self.recomputing_lock:
            self.bg_image = np.zeros( (self.height,self.width), dtype=np.uint8 )

        #self.OnClearBg(None)

    def offline_startup_func(self,arg):
        """gets called by fview_replay_fmf"""

        # automatically recompute mask and enable processing in offline mode
        self.recompute_mask = True # fire event
        assert self.mask_dirty==False
        self.processing_enabled = True

    def set_all_fview_plugins(self,plugins):
        print 'set_all_fview_plugins called'
        for plugin in plugins:
            print '  ', plugin.get_plugin_name()

            if plugin.get_plugin_name()=='FView external trigger':
                self.timestamp_modeler = plugin.timestamp_modeler
        print

    def quit(self):
        # save current maskdata
        pickle.dump(self.maskdata,open(self.pkl_fname,mode='w'))

        if self.current_stimulus_plugin is not None:
            self.current_stimulus_plugin.shutdown()

if __name__=='__main__':
    data = MaskData()
    data.configure_traits()
