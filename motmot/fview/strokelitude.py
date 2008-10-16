from __future__ import division

import pkg_resources

import wx
import wx.xrc as xrc

import numpy as np
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor
import cairo

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

class StrokelitudeClass:

    def __init__(self,wx_parent):
        self.frame = RES.LoadFrame(wx_parent,"FVIEW_STROKELITUDE_FRAME")
        self.draw_mask_ctrl = xrc.XRCCTRL(self.frame,'DRAW_MASK_REGION')
        self.maskdata = MaskData()
        self.maskdata.on_trait_change( self.on_mask_change )

        panel = xrc.XRCCTRL(self.frame,'TRAITS_PANEL')
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        control = self.maskdata.edit_traits( parent=panel,
                                             kind='subpanel',
                                             ).control
        sizer.Add(control, 1, wx.EXPAND)
        control.GetParent().SetMinSize(control.GetMinSize())

        self.on_mask_change() # initialize masks
        self.frame.Fit()


    def on_mask_change(self):
        left_quads = self.maskdata.get_quads('left')
        right_quads = self.maskdata.get_quads('right')

        ## for quad in left_quads:
        ##     print quad

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
        self.maskdata.maxx = max_width
        self.maskdata.maxy = max_height

if __name__=='__main__':

    data = MaskData()
    data.configure_traits()

