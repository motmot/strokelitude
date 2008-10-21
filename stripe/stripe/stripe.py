from __future__ import division
import simple_panels.simple_panels as simple_panels
import time
import numpy as np
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor

R2D = 180.0/np.pi

class StripeClass(traits.HasTraits):
    gain = traits.Float(-1.0)
    offset = traits.Float(0.0)

    traits_view = View( Group( ( Item('gain'),
                                 Item('offset') )),
                        title = 'Closed Loop Stripe Fixation',
                        )
    def __init__(self):
        self.panel_height=4
        self.panel_width=11
        self.compute_width=12
        self.stripe_pos_radians = 0.0
        self.last_time = time.time()
        self.gain = -1.0 # in (radians per second) / radians
        self.offset = 0.0 # in radians per second
        self._arr = np.zeros( (self.panel_height*8, self.panel_width*8),
                              dtype=np.uint8)
    def process_data( self, left_angle_radians, right_angle_radians ):
        #print 'left, right',left_angle_radians,right_angle_radians
        now = time.time()
        dt = now-self.last_time
        self.last_time = now

        L = left_angle_radians*R2D
        R = right_angle_radians*R2D
        diff_radians = left_angle_radians + right_angle_radians # (opposite signs already from angle measurement)
        vel = diff_radians*self.gain + self.offset
        #  print 'L = %.1f, R = %.1f, diff (L-R) = %.1f (degrees), velocity = %.1f (degrees/sec)'%(L,R,diff_radians*R2D, vel*R2D)
        self.stripe_pos_radians += vel*dt
        self.draw_stripe()
    def draw_stripe(self):
        self._arr.fill( 255 )
        #print 'self.stripe_pos 1',self.stripe_pos_radians*R2D
        self.stripe_pos_radians = self.stripe_pos_radians % (2*np.pi)
        #print 'self.stripe_pos 2',self.stripe_pos_radians*R2D
        pix_center = self.stripe_pos_radians/(2*np.pi)*(self.compute_width*8)
        #print 'pix_center',pix_center
        pix_width = 4
        pix_start = int(pix_center-pix_width/2.0)
        pix_stop = pix_start+pix_width

        pix_start = pix_start % (self.compute_width*8)
        pix_stop = pix_stop % (self.compute_width*8)

        if pix_start >= self._arr.shape[1]:
            pix_start = self._arr.shape[1]-1
        if pix_stop >= self._arr.shape[1]:
            pix_stop = self._arr.shape[1]
        self._arr[:,pix_start:pix_stop]=0
        #print pix_start
        simple_panels.display_frame(self._arr)

def rotate_stripe(revs=2,seconds_per_rev=2,fps=50):
    s = StripeClass()
    dt = 1.0/fps
    theta = np.linspace(0, revs*2*np.pi, revs*seconds_per_rev*fps )
    now = time.time()

    # XXX this is not a good way to do 'constant' velocity, since
    # errors will be cumulative.

    for i in range(len(theta)):
        expected_time = i*dt
        cur_time = time.time()-now
        #print '%d (should be %.1f) %.1f sec'%(i,expected_time,cur_time)
        s.stripe_pos_radians = theta[i]
        #print theta[i]*R2D
        s.draw_stripe()

        expected_time = i*dt
        sleep_time = expected_time - cur_time
        #print 'sleep_time',sleep_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        #print

if __name__=='__main__':
    #rotate_stripe(revs=1,seconds_per_rev=5)
    rotate_stripe(revs=10,seconds_per_rev=5)
