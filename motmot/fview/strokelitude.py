from __future__ import division

import pkg_resources

import wx
import wx.xrc as xrc

import numpy as np
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor

#from enthought.chaco2 import api as chaco2

# trigger extraction
RESFILE = pkg_resources.resource_filename(__name__,"fview_strokelitude.xrc")
RES = xrc.EmptyXmlResource()
RES.LoadFromString(open(RESFILE).read())

D2R = np.pi/180.0

class MaskPattern:
    """Describe the geometry of the pattern"""
    def __init__(self,x,y,r1,r2,alpha,beta,gamma):
        """initialize the state variables of the mask"""
        self.x = x
        self.y = y

        self.r1 = r1
        self.r2 = r2

        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma

    def get_all_linesegs(self,res=32):
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

        theta1 = np.linspace(alpha,beta,res) # left wing
        theta2 = np.linspace(-alpha,-beta,res) # right wing

        rotation = np.array([[ np.cos( gamma ), -np.sin(gamma)],
                             [ np.sin( gamma ), np.cos(gamma)]])
        translation = np.array( [[self.x],
                                 [self.y]], dtype=np.float64 )

        linesegs = []
        for theta in (theta1, theta2):
            # inner radius
            inner = np.array([self.r1*np.cos(theta),
                              self.r1*np.sin(theta)])
            # outer radius
            outer = np.array([self.r2*np.cos(theta[::-1]),
                              self.r2*np.sin(theta[::-1])])

            wing_verts = np.hstack(( inner, outer, inner[:,np.newaxis,0] ))

            wing_verts = np.dot(rotation, wing_verts) + translation
            linesegs.append( wing_verts.T.ravel() )

        return linesegs


class MaskData(traits.HasTraits):
    x = traits.Range(0,640,422)
    y = traits.Range(0,480,292)

    traits_view = View( Group( ( Item('x'),#,editor=RangeEditor()),
                                 Item('y'),#,editor=RangeEditor()),
                                 ),
                               orientation = 'horizontal',
                               show_border = False,
                               ),
                        title = 'Mask Parameters' )

class StrokelitudeClass:

    def __init__(self,wx_parent):
        self.wx_parent = wx_parent
        self.frame = RES.LoadFrame(self.wx_parent,"FVIEW_STROKELITUDE_FRAME")
        self._init_frame()
        self.masks = {}

    def _init_frame(self):
        panel = xrc.XRCCTRL(self.frame,'TRAITS_PANEL')
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        data = MaskData()
        control = data.edit_traits( parent=panel,
                                    kind='subpanel',
                                    ).control
        sizer.Add(control, 1, wx.EXPAND)
        control.GetParent().SetMinSize(control.GetMinSize())

        self.frame.Fit()

        # bind controllers
        ctrl = xrc.XRCCTRL(self.frame,'CENTER_X')
        wx.EVT_COMMAND_SCROLL(ctrl, ctrl.GetId(), self.OnCenterX)
        ctrl = xrc.XRCCTRL(self.frame,'CENTER_Y')
        wx.EVT_COMMAND_SCROLL(ctrl, ctrl.GetId(), self.OnCenterY)
        ctrl = xrc.XRCCTRL(self.frame,'ARC_R1')
        wx.EVT_COMMAND_SCROLL(ctrl, ctrl.GetId(), self.OnArcR1)
        ctrl = xrc.XRCCTRL(self.frame,'ARC_R2')
        wx.EVT_COMMAND_SCROLL(ctrl, ctrl.GetId(), self.OnArcR2)
        ctrl = xrc.XRCCTRL(self.frame,'ARC_ALPHA')
        wx.EVT_COMMAND_SCROLL(ctrl, ctrl.GetId(), self.OnArcAlpha)
        ctrl = xrc.XRCCTRL(self.frame,'ARC_BETA')
        wx.EVT_COMMAND_SCROLL(ctrl, ctrl.GetId(), self.OnArcBeta)
        ctrl = xrc.XRCCTRL(self.frame,'ARC_GAMMA')
        wx.EVT_COMMAND_SCROLL(ctrl, ctrl.GetId(), self.OnArcGamma)

        self.draw_mask_ctrl = xrc.XRCCTRL(self.frame,'DRAW_MASK_REGION')


    def OnCenterX(self, event):
        ctrl = xrc.XRCCTRL(self.frame,'CENTER_X')
        for mask in self.masks.itervalues():
            mask.x = ctrl.GetValue()
    def OnCenterY(self, event):
        ctrl = xrc.XRCCTRL(self.frame,'CENTER_Y')
        for mask in self.masks.itervalues():
            mask.y = ctrl.GetValue()
    def OnArcR1(self, event):
        ctrl_r1 = xrc.XRCCTRL(self.frame,'ARC_R1')
        ctrl_r2 = xrc.XRCCTRL(self.frame,'ARC_R2')
        v1 = ctrl_r1.GetValue()
        if v1 > ctrl_r2.GetValue():
            ctrl_r1.SetValue( ctrl_r2.GetValue() )
        else:
            for mask in self.masks.itervalues():
                mask.r1 = v1

    def OnArcR2(self, event):
        ctrl_r1 = xrc.XRCCTRL(self.frame,'ARC_R1')
        ctrl_r2 = xrc.XRCCTRL(self.frame,'ARC_R2')
        v2 = ctrl_r2.GetValue()
        if ctrl_r1.GetValue() > v2:
            ctrl_r2.SetValue( ctrl_r1.GetValue() )
        else:
            for mask in self.masks.itervalues():
                mask.r2 = v2

    def OnArcAlpha(self, event):
        ctrl = xrc.XRCCTRL(self.frame,'ARC_ALPHA')
        for mask in self.masks.itervalues():
            mask.alpha = ctrl.GetValue()
    def OnArcGamma(self, event):
        ctrl = xrc.XRCCTRL(self.frame,'ARC_GAMMA')
        for mask in self.masks.itervalues():
            mask.gamma = ctrl.GetValue()
    def OnArcBeta(self, event):
        ctrl = xrc.XRCCTRL(self.frame,'ARC_BETA')
        for mask in self.masks.itervalues():
            mask.beta = ctrl.GetValue()

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
            mask = self.masks[cam_id]
            draw_linesegs.extend( mask.get_all_linesegs() )
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

        self.masks[cam_id] = MaskPattern( xrc.XRCCTRL(self.frame,'CENTER_X').GetValue(),
                                          xrc.XRCCTRL(self.frame,'CENTER_Y').GetValue(),

                                          xrc.XRCCTRL(self.frame,'ARC_R1').GetValue(),
                                          xrc.XRCCTRL(self.frame,'ARC_R2').GetValue(),

                                          xrc.XRCCTRL(self.frame,'ARC_ALPHA').GetValue(),
                                          xrc.XRCCTRL(self.frame,'ARC_BETA').GetValue(),
                                          xrc.XRCCTRL(self.frame,'ARC_GAMMA').GetValue(),
                                          )


if __name__=='__main__':

    data = MaskData()
    data.configure_traits()

