from __future__ import division

import pkg_resources

import wx
import wx.xrc as xrc

import numpy as np
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor
import cairo
import scipy.sparse

from enthought.enable2.api import Component, Container
from enthought.enable2.wx_backend.api import Window

#from enthought.chaco2 import api as chaco2

# trigger extraction
RESFILE = pkg_resources.resource_filename(__name__,"fview_strokelitude.xrc")
RES = xrc.EmptyXmlResource()
RES.LoadFromString(open(RESFILE).read())

D2R = np.pi/180.0


class MaskData(traits.HasTraits):
    x = traits.Float(422.0)
    y = traits.Float(292.0)
    wingsplit = traits.Float(20.0)
    r1 = traits.Float(100.0)
    r2 = traits.Float(151.0)

    alpha = traits.Range(0.0, 180.0, 52.0)
    beta = traits.Range(0.0, 180.0, 82.0)
    gamma = traits.Range(0.0, 360.0, 206.0)

    # these are just necesary for establishing limits in the view:
    maxx = traits.Float(699.9)
    maxy = traits.Float(699.9)
    maxdim = traits.Float(500)

    def _maxx_changed(self):
        self.maxdim = np.sqrt(self.maxx**2+self.maxy**2)
    def _maxy_changed(self):
        self.maxdim = np.sqrt(self.maxx**2+self.maxy**2)

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

    def _get_rotation_translation(self):
        gamma = self.gamma*D2R
        rotation = np.array([[ np.cos( gamma ), -np.sin(gamma)],
                             [ np.sin( gamma ), np.cos(gamma)]])
        translation = np.array( [[self.x],
                                 [self.y]], dtype=np.float64 )
        return rotation,translation

    def get_extra_linesegs(self):
        """return linesegments that contextualize parameters"""
        linesegs = []
        rotation,translation = self._get_rotation_translation()
        if 1:
            # longitudinal axis (along fly's x coord)
            verts = np.array([[-100,100],
                              [0,     0]],dtype=np.float)
            verts = np.dot(rotation, verts) + translation
            linesegs.append( verts.T.ravel() )
        if 1:
            # transverse axis (along fly's y coord)
            verts = np.array([[0,0],
                              [-10,10]],dtype=np.float)
            verts = np.dot(rotation, verts) + translation
            linesegs.append( verts.T.ravel() )
        return linesegs

    def get_quads(self,side,res=5):
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

        if side=='left':
            all_theta = np.linspace(alpha,beta,res+1)
            wingsplit_trans = np.array([[0.0],[self.wingsplit]])
        elif side=='right':
            all_theta = np.linspace(-alpha,-beta,res+1)
            wingsplit_trans = np.array([[0.0],[-self.wingsplit]])

        rotation,translation = self._get_rotation_translation()

        linesegs = []
        for i in range(res):
            theta = all_theta[i:(i+2)]
            # inner radius
            inner = np.array([self.r1*np.cos(theta),
                              self.r1*np.sin(theta)])
            # outer radius
            outer = np.array([self.r2*np.cos(theta[::-1]),
                              self.r2*np.sin(theta[::-1])])

            wing_verts = np.hstack(( inner, outer, inner[:,np.newaxis,0] ))+wingsplit_trans

            wing_verts = np.dot(rotation, wing_verts) + translation
            linesegs.append( wing_verts.T.ravel() )

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
    if 1:
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

class Box(Component):
    resizable = ""

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        gc.save_state()
        gc.set_fill_color((1.0, 0.0, 0.0, 1.0))
        dx, dy = self.bounds
        x, y = self.position
        gc.rect(x, y, dx, dy)
        gc.fill_path()
        gc.restore_state()


class StrokelitudeClass(traits.HasTraits):
    mask_dirty = traits.Bool(True) # True the mask parameters changed

    def _mask_dirty_changed(self):
        if self.mask_dirty:
            self.recompute_mask_button.Enable(True)
        else:
            self.recompute_mask_button.Enable(False)

    def __init__(self,wx_parent):
        self.frame = RES.LoadFrame(wx_parent,"FVIEW_STROKELITUDE_FRAME")
        self.draw_mask_ctrl = xrc.XRCCTRL(self.frame,'DRAW_MASK_REGION')
        self.maskdata = MaskData()
        self.maskdata.on_trait_change( self.on_mask_change )

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

        if 1:
            box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
            component = box

            panel = xrc.XRCCTRL(self.frame,'LIVEVIEW_PANEL')
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

        ## ID_Timer2 = wx.NewId()
        ## self.timer2 = wx.Timer(self.frame, ID_Timer2)
        ## wx.EVT_TIMER(self, ID_Timer2, self.OnTimer2)
        ## self.timer2.Start(100)

    def recompute_mask(self,event):
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

        print left_mat.shape
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

        if self.draw_mask_ctrl.IsChecked():
            draw_linesegs.extend( self.maskdata.get_quads('left') )
            draw_linesegs.extend( self.maskdata.get_quads('right') )
            draw_linesegs.extend( self.maskdata.get_extra_linesegs() )

        enabled = True
        if enabled and not self.mask_dirty:

            this_image = np.asarray(buf)
            this_image_flat = this_image.ravel()

            left_vals  = self.left_mat_sparse  * this_image_flat
            right_vals = self.right_mat_sparse * this_image_flat
            #print 'left,right',left_vals,right_vals

        return draw_points, draw_linesegs

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

if __name__=='__main__':

    data = MaskData()
    data.configure_traits()

