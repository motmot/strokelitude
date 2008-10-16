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
    x = traits.Range(0.0, 640.0, 422.0)
    y = traits.Range(0.0, 480.0, 292.0)
    wingsplit = traits.Range(0.0, 640.0, 20.0)
    r1 = traits.Range(0.0, 640.0, 100.0)
    r2 = traits.Range(0.0, 640.0, 151.0)
    alpha = traits.Range(0.0, 180.0, 52.0)
    beta = traits.Range(0.0, 180.0, 82.0)
    gamma = traits.Range(0.0, 360.0, 206.0)

    traits_view = View( Group( ( Item('x',
                                      #editor=RangeEditor(), # broken?
                                      ),
                                 Item('y'),
                                 Item('wingsplit'),
                                 Item('r1'),
                                 Item('r2'),
                                 Item('alpha'),
                                 Item('beta'),
                                 Item('gamma'),
                                 ),
                               orientation = 'horizontal',
                               show_border = False,
                               ),
                        title = 'Mask Parameters',
                        )

    def get_rectangles(self,side,res=5):
        """return linesegments outlining the pattern (for OpenGL type display)

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
        gamma = self.gamma*D2R

        if side=='left':
            all_theta = np.linspace(alpha,beta,res+1)
            wingsplit_trans = np.array([[0.0],[self.wingsplit]])
        elif side=='right':
            all_theta = np.linspace(-alpha,-beta,res+1)
            wingsplit_trans = np.array([[0.0],[-self.wingsplit]])

        rotation = np.array([[ np.cos( gamma ), -np.sin(gamma)],
                             [ np.sin( gamma ), np.cos(gamma)]])
        translation = np.array( [[self.x],
                                 [self.y]], dtype=np.float64 )

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

        panel = xrc.XRCCTRL(self.frame,'TRAITS_PANEL')
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        control = self.maskdata.edit_traits( parent=panel,
                                             kind='subpanel',
                                             ).control
        sizer.Add(control, 1, wx.EXPAND)
        control.GetParent().SetMinSize(control.GetMinSize())

        self.frame.Fit()

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
            draw_linesegs.extend( self.maskdata.get_rectangles('left') )
            draw_linesegs.extend( self.maskdata.get_rectangles('right') )
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
        pass

if __name__=='__main__':

    data = MaskData()
    data.configure_traits()

