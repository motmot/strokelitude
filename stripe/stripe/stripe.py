from __future__ import division
import os, sys
if int(os.environ.get('DISABLE_PANELS','0')):
    simple_panels = None
else:
    import simple_panels.simple_panels as simple_panels
import time
import numpy as np
import Queue
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor

import remote_traits

R2D = 180.0/np.pi

class StripeClass(remote_traits.MaybeRemoteHasTraits):
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
        self.vel = 0.0
        self.last_diff_radians = 0.0
        self.incoming_data_queue = Queue.Queue()

    def do_work( self ):
        """This gets called frequently (e.g. 100 Hz)"""

        # Get any available incoming data. Ignore all but most recent.
        last_data = None
        while 1:
            try:
                last_data = self.incoming_data_queue.get_nowait()
            except Queue.Empty, err:
                break

        # Update stripe velocity if new data arrived.
        if last_data is not None:
            (cam_id,timestamp,framenumber,results) = last_data
            left_angle_radians, right_angle_radians = results
            #L = left_angle_radians*R2D
            #R = right_angle_radians*R2D
            diff_radians = left_angle_radians + right_angle_radians # (opposite signs already from angle measurement)
            self.last_diff_radians = diff_radians

        # update self.vel every frame so that changes in gain and offset noticed
        self.vel = self.last_diff_radians*self.gain + self.offset

        # Compute stripe position.
        now = time.time()
        dt = now-self.last_time
        self.last_time = now

        self.stripe_pos_radians += self.vel*dt

        # Draw stripe
        self.draw_stripe()

    def draw_stripe(self):
        self._arr.fill( 255 ) # turn all pixels white

        # compute columns of stripe
        self.stripe_pos_radians = self.stripe_pos_radians % (2*np.pi)
        pix_center = self.stripe_pos_radians/(2*np.pi)*(self.compute_width*8)
        pix_width = 4
        pix_start = int(pix_center-pix_width/2.0)
        pix_stop = pix_start+pix_width

        pix_start = pix_start % (self.compute_width*8)
        pix_stop = pix_stop % (self.compute_width*8)

        if pix_start >= self._arr.shape[1]:
            pix_start = self._arr.shape[1]-1
        if pix_stop >= self._arr.shape[1]:
            pix_stop = self._arr.shape[1]

        # make the stripe pixels black
        self._arr[:,pix_start:pix_stop]=0

        if simple_panels is not None:
            # send to USB
            simple_panels.display_frame(self._arr)
        else:
            sys.stdout.write('%d '%round(pix_center))
            sys.stdout.flush()

def rotate_stripe(revs=2,seconds_per_rev=2,fps=50):
    s = StripeClass()
    dt = 1.0/fps
    theta = np.linspace(0, revs*2*np.pi, revs*seconds_per_rev*fps )
    now = time.time()

    for i in range(len(theta)):
        expected_time = i*dt
        cur_time = time.time()-now
        s.stripe_pos_radians = theta[i]
        s.draw_stripe()

        expected_time = i*dt
        sleep_time = expected_time - cur_time
        if sleep_time > 0:
            time.sleep(sleep_time)

DO_HOSTNAME='localhost'
DO_PORT = 8442

def mainloop():
    stripe = StripeClass()
    server = remote_traits.ServerObj(DO_HOSTNAME,DO_PORT)
    server.serve_name('stripe',stripe)

    desired_rate = 500.0 #hz
    dt = 1.0/desired_rate

    while 1: # run forever
        server.handleRequests(timeout=dt) # handle network events
        stripe.do_work()

if __name__=='__main__':
    #rotate_stripe(revs=1,seconds_per_rev=5)
    #rotate_stripe(revs=10,seconds_per_rev=5)
    mainloop()
