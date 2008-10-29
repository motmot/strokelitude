from __future__ import division, with_statement

import pkg_resources

import wx
import wx.xrc as xrc
import warnings, time, threading

import numpy as np
import cairo
import scipy.sparse
import Queue
import os

import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor

from enthought.enable2.api import Component, Container
from enthought.enable2.wx_backend.api import Window
from enthought.chaco2.api import DataView, ArrayDataSource, ScatterPlot, LinePlot, LinearMapper
from enthought.chaco2.api import create_line_plot, add_default_axes, add_default_grids

# trigger extraction
RESFILE = pkg_resources.resource_filename(__name__,"fview_strokelitude.xrc")
RES = xrc.EmptyXmlResource()
RES.LoadFromString(open(RESFILE).read())

DataReadyEvent = wx.NewEventType()

D2R = np.pi/180.0

def load_plugins():
    # modified from motmot.fview.plugin_manager
    PluginClasses = []
    loaded_components = []
    pkg_env = pkg_resources.Environment()
    for name in pkg_env:
        egg = pkg_env[name][0]
        modules = []

        for name in egg.get_entry_map('motmot.fview_strokelitude.plugins'):
            egg.activate()
            entry_point = egg.get_entry_info('motmot.fview_strokelitude.plugins', name)
            if entry_point.module_name not in loaded_components:
                try:
                    PluginClass = entry_point.load()
                except Exception,x:
                    if int(os.environ.get('FVIEW_RAISE_ERRORS','0')):
                        raise
                    else:
                        import warnings
                        warnings.warn('could not load plugin (set env var FVIEW_RAISE_ERRORS to raise error) %s: %s'%(str(entry_point),str(x)))
                        continue
                PluginClasses.append( PluginClass )
                modules.append(entry_point.module_name)
                loaded_components.append(entry_point.module_name)
    # make instances of plugins
    plugins = [PluginClass() for PluginClass in PluginClasses]
    return plugins

class MaskData(traits.HasTraits):
    # lengths, in pixels
    x = traits.Float(401.0)
    y = traits.Float(273.5)
    wingsplit = traits.Float(70.3)
    r1 = traits.Float(22.0)
    r2 = traits.Float(80.0)

    # angles, in degrees
    alpha = traits.Range(0.0, 180.0, 14.0)
    beta = traits.Range(0.0, 180.0, 87.0)
    gamma = traits.Range(0.0, 360.0, 206.0)

    # number of angular bins
    nbins = traits.Int(30)

    # these are just necesary for establishing limits in the view:
    maxx = traits.Float(699.9)
    maxy = traits.Float(699.9)
    maxdim = traits.Float(500)

    def _maxx_changed(self):
        self.maxdim = np.sqrt(self.maxx**2+self.maxy**2)
    def _maxy_changed(self):
        self.maxdim = np.sqrt(self.maxx**2+self.maxy**2)

    def _wingsplit_changed(self):
        self._wingsplit_translation['left'] = np.array([[0.0],[self.wingsplit]])
        self._wingsplit_translation['right'] = np.array([[0.0],[-self.wingsplit]])

    def _gamma_changed(self):
        gamma = self.gamma*D2R
        self._rotation = np.array([[ np.cos( gamma ), -np.sin(gamma)],
                                   [ np.sin( gamma ), np.cos(gamma)]])

    def _xy_changed(self):
        self._translation = np.array( [[self.x],
                                       [self.y]],
                                      dtype=np.float64 )

    # If either self.x or self.y changed, call _xy_changed()
    _x_changed = _xy_changed
    _y_changed = _xy_changed

    def _alpha_beta_nbins_changed(self):
        alpha = self.alpha*D2R
        beta = self.beta*D2R
        self._all_theta['left'] = np.linspace(alpha,beta,self.nbins+1)
        self._all_theta['right'] = np.linspace(-alpha,-beta,self.nbins+1)

    # If any of alpha, beta or nbins changed
    _alpha_changed = _alpha_beta_nbins_changed
    _beta_changed = _alpha_beta_nbins_changed
    _nbins_changed = _alpha_beta_nbins_changed

    def __init__(self):
        self._wingsplit_translation = {}
        self._all_theta = {}
        self._wingsplit_changed()
        self._gamma_changed()
        self._xy_changed()
        self._alpha_beta_nbins_changed()

    traits_view = View( Group( ( Item('x',
                                      editor=RangeEditor(high_name='maxx',
                                                         format='%.1f',
                                                         label_width=50,
                                                         ),
                                      ),
                                 Item('y',
                                      editor=RangeEditor(high_name='maxy',
                                                         format='%.1f',
                                                         label_width=50,
                                                         ),
                                      ),
                                 Item('wingsplit',
                                      editor=RangeEditor(high_name='maxdim',
                                                         format='%.1f',
                                                         label_width=50,
                                                         ),
                                      ),
                                 Item('r1',
                                      editor=RangeEditor(high_name='r2',
                                                         format='%.1f',
                                                         label_width=50,
                                                         ),
                                      ),
                                 Item('r2',
                                      editor=RangeEditor(low_name='r1',
                                                         high_name='maxdim',
                                                         format='%.1f',
                                                         label_width=50,
                                                         ),
                                      ),
                                 Item('alpha'),
                                 Item('beta'),
                                 Item('gamma'),
                                 ),
                               orientation = 'horizontal',
                               show_border = False,
                               ),
                        title = 'Mask Parameters',
                        )

    def get_extra_linesegs(self):
        """return linesegments that contextualize parameters"""
        linesegs = []
        if 1:
            # longitudinal axis (along fly's x coord)
            verts = np.array([[-100,100],
                              [0,     0]],dtype=np.float)
            verts = np.dot(self._rotation, verts) + self._translation
            linesegs.append( verts.T.ravel() )
        if 1:
            # transverse axis (along fly's y coord)
            verts = np.array([[0,0],
                              [-10,10]],dtype=np.float)
            verts = np.dot(self._rotation, verts) + self._translation
            linesegs.append( verts.T.ravel() )
        if 1:
            # half-circle centers
            n = 10
            theta = np.linspace(0,2*np.pi,10)
            verts = np.array([ 5*np.cos(theta),
                               5*np.sin(theta) ])
            vleft = verts+self._wingsplit_translation['left']
            vright = verts+self._wingsplit_translation['right']

            vleft = np.dot(self._rotation, vleft) + self._translation
            vright = np.dot(self._rotation, vright) + self._translation

            linesegs.append( vleft.T.ravel() )
            linesegs.append( vright.T.ravel() )

        if 1:
            # left wing
            verts = np.array([[0.0, 0.0, 20.0],
                              [20.0,0.0, 0.0]])
            vleft = verts+self._wingsplit_translation['left']

            vleft = np.dot(self._rotation, vleft) + self._translation

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

        all_theta = self._all_theta[side]

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
            wing_verts += self._wingsplit_translation[side]

            wing_verts = np.dot(self._rotation, wing_verts) + self._translation
            linesegs.append( wing_verts.T.ravel() )

        return linesegs

    def index2angle(self,side,idx):
        """convert index to angle (in radians)"""
        alpha = self.alpha*D2R
        beta = self.beta*D2R
        if side=='right':
            alpha = -alpha
            beta = -beta
        frac = idx/self.nbins
        diff = beta-alpha
        return alpha+frac*diff

    def get_span_lineseg(self,side,theta):
        """draw line on side at angle theta (in radians)"""
        linesegs = []

        verts = np.array( [[ 0, 1000.0*np.cos(theta)],
                           [ 0, 1000.0*np.sin(theta)]] )
        verts = verts + self._wingsplit_translation[side]
        verts = np.dot(self._rotation, verts) + self._translation
        linesegs.append( verts.T.ravel() )
        return linesegs

def quad2imvec(quad,width,height,debug_count=0):
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

    for (x,y) in zip(quad[2::2],
                     quad[3::2]):
        ctx.line_to(x,y)
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
    arr = arr.astype(np.float64)
    arr = arr/255.0 # max value should be 1.0
    imvec = arr.ravel()
    return imvec

def resample_sparse( orig, orig_h, orig_w, offset, new_h, new_w ):
    print 'resampling'
    orig_dense = orig.todense()
    print 'type(orig_dense)',type(orig_dense)
    orig_dense = np.asarray(orig_dense) # convert to np.array (from matrix)
    print 'type(orig_dense)',type(orig_dense)
    print 'orig_dense.shape',orig_dense.shape

    new_dense = []
    xoff,yoff = offset
    for orig_row in orig_dense:
        orig_mat = np.reshape( orig_row, (orig_h,orig_w) )
        print 'orig_mat.shape',orig_mat.shape
        resampled_mat = orig_mat[ yoff:(yoff+new_h), xoff:(xoff+new_w) ]
        print 'resampled_mat.shape',resampled_mat.shape
        imvec = resampled_mat.ravel()
        print 'imvec.shape',imvec.shape
        new_dense.append( imvec )
    new_dense = np.array(new_dense)
    result = scipy.sparse.csc_matrix(new_dense)
    return result

class StrokelitudeClass(traits.HasTraits):
    """plugin for fview.

    Because this class implements everything necessary to be a valid fview plugin,
    some of the methods here are necessary.
    """
    mask_dirty = traits.Bool(True) # True the mask parameters changed

    def _mask_dirty_changed(self):
        if self.mask_dirty:
            self.recompute_mask_button.Enable(True)
        else:
            self.recompute_mask_button.Enable(False)

    def __init__(self,wx_parent):
        self.wx_parent = wx_parent

        self.frame = RES.LoadFrame(wx_parent,"FVIEW_STROKELITUDE_FRAME")
        self.draw_mask_ctrl = xrc.XRCCTRL(self.frame,'DRAW_MASK_REGION')
        self.maskdata = MaskData()
        self.maskdata.on_trait_change( self.on_mask_change )
        self.vals_queue = Queue.Queue()
        self.bg_queue = Queue.Queue()

        self.recomputing_lock = threading.Lock()

        if 1:
            # load plugins
            self.plugins = load_plugins()
            print 'fview_strokelitude loaded plugins:',self.plugins
            if len(self.plugins)>1:
                warnings.warn('currently only support for max 1 plugin')
                del self.plugins[1:]

            if len(self.plugins):
                plugin = self.plugins[0]

                panel = xrc.XRCCTRL(self.frame,'PLUGIN_PANEL')
                sizer = wx.BoxSizer(wx.HORIZONTAL)

                control = plugin.edit_traits( parent=panel,
                                              kind='subpanel',
                                              ).control
                sizer.Add(control, 1, wx.EXPAND)
                panel.SetSizer( sizer )
                control.GetParent().SetMinSize(control.GetMinSize())


        if 1:
            # setup maskdata parameter panel
            panel = xrc.XRCCTRL(self.frame,'MASKDATA_PARAMS_PANEL')
            sizer = wx.BoxSizer(wx.HORIZONTAL)

            control = self.maskdata.edit_traits( parent=panel,
                                                 kind='subpanel',
                                                 ).control
            sizer.Add(control, 1, wx.EXPAND)
            panel.SetSizer( sizer )
            control.GetParent().SetMinSize(control.GetMinSize())

            for side in ['left','right']:

                x=np.arange(self.maskdata.nbins,dtype=np.float64)
                y=np.zeros_like(x)
                plot = create_line_plot((x,y), color="red", width=2.0,
                                        border_visible=True,
                                        #add_axis=True,
                                        )
                value_range = plot.value_mapper.range
                index_range = plot.index_mapper.range

                plot.padding = 10
                plot.bgcolor = "white"

                if side=='left':
                    self.left_plot=plot
                elif side=='right':
                    self.right_plot=plot

                component = plot

                panel = xrc.XRCCTRL(self.frame,side.upper()+'_LIVEVIEW_PANEL')
                sizer = wx.BoxSizer(wx.HORIZONTAL)
                self.enable_window = Window( panel, -1, component = component )
                control = self.enable_window.control
                sizer.Add(control, 1, wx.EXPAND)
                panel.SetSizer( sizer )
                control.GetParent().SetMinSize(control.GetMinSize())

        self.cam_id = None
        self.width = 20
        self.height = 10

        self.frame.Fit()
        self.mask_dirty=True

        self.recompute_mask_button = xrc.XRCCTRL(self.frame,'RECOMPUTE_MASK')
        wx.EVT_BUTTON(self.recompute_mask_button, self.recompute_mask_button.GetId(),
                      self.recompute_mask)

        ctrl = xrc.XRCCTRL(self.frame,'TAKE_BG_BUTTON')
        wx.EVT_BUTTON(ctrl, ctrl.GetId(), self.OnTakeBg)
        ctrl = xrc.XRCCTRL(self.frame,'CLEAR_BG_BUTTON')
        wx.EVT_BUTTON(ctrl, ctrl.GetId(), self.OnClearBg)

        self.enabled_box = xrc.XRCCTRL(self.frame,'ENABLE_PROCESSING')

        self.frame.Connect( -1, -1, DataReadyEvent, self.OnDataReady )

    def recompute_mask(self,event):
        with self.recomputing_lock:
            count = 0

            left_quads = self.maskdata.get_quads('left')
            right_quads = self.maskdata.get_quads('right')

            left_mat = []
            for quad in left_quads:
                imvec = quad2imvec(quad,self.width,self.height,debug_count=count)
                left_mat.append( imvec )
                count+=1

            left_mat = np.array(left_mat)
            self.left_mat_sparse = scipy.sparse.csc_matrix(left_mat)

            right_mat = []
            for quad in right_quads:
                imvec = quad2imvec(quad,self.width,self.height,debug_count=count)
                right_mat.append( imvec )
                count+=1

            right_mat = np.array(right_mat)
            self.right_mat_sparse = scipy.sparse.csc_matrix(right_mat)

            self._sparse_roi_cache = None # clear cache
            self._recomputed_timestamp = time.time() # use as a hash value

            bg_image_flat = self.bg_image.ravel()
            self.bg_left_vec = self.left_mat_sparse  * bg_image_flat
            self.bg_right_vec = self.right_mat_sparse  * bg_image_flat

            self.mask_dirty=False

    def on_mask_change(self):
        self.mask_dirty=True

    def get_frame(self):
        """return wxPython frame widget"""
        return self.frame

    def get_plugin_name(self):
        return 'Stroke Amplitude'

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

        have_lock = self.recomputing_lock.acquire(False) # don't block
        if not have_lock:
            return draw_points, draw_linesegs
        try:
            if self.draw_mask_ctrl.IsChecked():
                # XXX this is naughty -- it's not threasafe.
                # concatenate lists
                extra = ( self.maskdata.get_quads('left') +
                          self.maskdata.get_quads('right') +
                          self.maskdata.get_extra_linesegs() )

                draw_linesegs.extend( extra )

            command = None
            while 1:
                try:
                    # all we care about is last command
                    command = self.bg_queue.get_nowait()
                except Queue.Empty, err:
                    break

            this_image = np.asarray(buf)
            if command == 'clear':
                self.bg_image = np.zeros_like( this_image )
                self.mask_dirty = True # naughty to cross thread boundary
            elif command == 'take':
                self.bg_image = np.array( this_image, copy=True )
                self.mask_dirty = True # naughty to cross thread boundary

            # XXX naughty to cross thread boundary to get enabled_box value, too
            if self.enabled_box.GetValue() and not self.mask_dirty:

                h,w = this_image.shape
                if not (self.width==w and self.height==h):
                    warnings.warn('no ROI support for calculating stroke amplitude')

                    tmp = self._sparse_roi_cache

                    cache_ok = False
                    if tmp is not None:
                        (old_timestamp, old_offset, old_wh, left, right) = tmp
                        if (old_timestamp == self._recomputed_timestamp and
                            old_offset == buf_offset and
                            old_wh == (w,h) ):

                            left_mat_sparse = left
                            right_mat_sparse = right
                            cache_ok = True
                    if not cache_ok:
                        left_mat_sparse = resample_sparse( self.left_mat_sparse,
                                                           self.height,
                                                           self.width,
                                                           buf_offset,
                                                           h,w )
                        right_mat_sparse = resample_sparse( self.right_mat_sparse,
                                                            self.height,
                                                            self.width,
                                                            buf_offset,
                                                            h,w )
                        self._sparse_roi_cache = ( self._recomputed_timestamp,
                                                   buf_offset,
                                                   (w,h),
                                                   left_mat_sparse,
                                                   right_mat_sparse )

                    raise NotImplementedError('need to support background images for ROI')

                else:
                    left_mat_sparse = self.left_mat_sparse
                    right_mat_sparse = self.right_mat_sparse

                    bg_left_vec = self.bg_left_vec
                    bg_right_vec = self.bg_right_vec

                this_image_flat = this_image.ravel()

                left_vals  = left_mat_sparse  * this_image_flat
                right_vals = right_mat_sparse * this_image_flat

                left_vals = left_vals - bg_left_vec
                right_vals = right_vals - bg_right_vec

                results = []
                for side in ('left','right'):
                    if side=='left':
                        vals = left_vals
                    else:
                        vals = right_vals

                    min_val = vals.min()
                    mid_val = (min_val + vals.max())/2
                    if min_val==mid_val:
                        continue
                    all_idxs = np.nonzero(vals>=mid_val)[0]
                    assert len(all_idxs) > 0
                    second_idx = all_idxs[0]
                    first_idx = second_idx - 1
                    if first_idx==-1:
                        interp_idx = 0.0
                    else:
                        # y = mx+b
                        m = vals[second_idx] - vals[first_idx] # slope (indices are unity apart)
                        b = vals[first_idx] - m*first_idx
                        interp_idx = (mid_val-b)/m

                    #latency_sec = time.time()-timestamp
                    #print 'msec % 5.1f'%(latency_sec*1000.0,)

                    angle_radians = self.maskdata.index2angle(side,
                                                              interp_idx)
                    results.append( angle_radians )

                    # draw lines
                    draw_linesegs.extend(
                        self.maskdata.get_span_lineseg(side,angle_radians))

                for plugin in self.plugins:
                    plugin.process_data( *results )

                # send values from each quad to be drawn
                self.vals_queue.put( (left_vals, right_vals) )
                event = wx.CommandEvent(DataReadyEvent)
                event.SetEventObject(self.frame)
                wx.PostEvent(self.frame, event) # triggers call to self.OnDataReady
        finally:
            self.recomputing_lock.release()

        return draw_points, draw_linesegs

    def OnTakeBg(self,event):
        self.bg_queue.put('take')

    def OnClearBg(self,event):
        self.bg_queue.put('clear')

    def OnDataReady(self, event):
        lrvals = None
        while 1:
            try:
                lrvals = self.vals_queue.get_nowait()
            except Queue.Empty:
                break
        if lrvals is None:
            return

        left_vals, right_vals = lrvals
        self.left_plot.value.set_data(left_vals)
        self.right_plot.value.set_data(right_vals)

    def set_view_flip_LR( self, val ):
        pass

    def set_view_rotate_180( self, val ):
        pass

    def quit(self):
        pass

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

        self.bg_image = np.zeros( (max_height, max_width),
                                  dtype=np.uint8)

if __name__=='__main__':

    data = MaskData()
    data.configure_traits()

