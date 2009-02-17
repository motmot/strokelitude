from __future__ import division
import os, sys
if int(os.environ.get('DISABLE_PANELS','0')):
    simple_panels = None
else:
    import simple_panels.simple_panels as simple_panels # try DISABLE_PANELS=1
import time, warnings
import numpy as np
import Queue
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor

import remote_traits
import strokelitude.plugin

R2D = 180.0/np.pi
D2R = np.pi/180.0

class StripePluginInfo(strokelitude.plugin.PluginBase):
    def get_name(self):
        return 'Closed Loop Stripe Fixation'
    def get_hastraits_class(self):
        return StripeClass, StripeClassWorker

class StripeClass(remote_traits.MaybeRemoteHasTraits):
    gain = traits.Float(-900.0) # in (degrees per second) / degrees
    offset = traits.Float(0.0)  # in degrees per second

    traits_view = View( Group( ( Item(name='gain',label='gain [ (deg/sec) / deg ]'),
                                 Item(name='offset',label='offset [ deg/sec ]') )),
                        )

class StripeClassWorker(StripeClass):
    def __init__(self,display_text_queue=None):
        super(StripeClassWorker,self).__init__()
        self.display_text_queue=display_text_queue
        self.panel_height=4
        self.panel_width=11
        self.compute_width=12
        self.stripe_pos_degrees = 0.0
        self.last_time = time.time()
        self.arr = np.zeros( (self.panel_height*8, self.panel_width*8),
                              dtype=np.uint8)
        self.vel = 0.0
        self.last_diff_degrees = 0.0
        self.incoming_data_queue = Queue.Queue()
        if simple_panels is not None:
            self.panel_device = simple_panels.DumpframeDevice()
        else:
            self.panel_device = None

    def _gain_changed(self):
        self.display_text_queue.put('gain %.1f'%self.gain)

    def _offset_changed(self):
        self.display_text_queue.put('offset %.1f'%self.offset)

    def set_incoming_queue(self,data_queue):
        self.incoming_data_queue = data_queue

    def do_work( self ):
        """This gets called frequently (e.g. 100 Hz)"""

        ## # Get any available incoming data. Ignore all but most recent.
        last_data = None
        while 1:
            try:
                last_data = self.incoming_data_queue.get_nowait()
            except Queue.Empty, err:
                break

        # Update stripe velocity if new data arrived.
        if last_data is not None:
            (framenumber, left_angle_degrees, right_angle_degrees,
             trigger_timestamp) = last_data
            if not (np.isnan(left_angle_degrees) or np.isnan(right_angle_degrees)):
                diff_degrees = right_angle_degrees - left_angle_degrees
                self.last_diff_degrees = diff_degrees

        # update self.vel every frame so that changes in gain and offset noticed
        self.vel = self.last_diff_degrees*self.gain + self.offset

        # Compute stripe position.
        now = time.time()
        dt = now-self.last_time
        self.last_time = now

        self.stripe_pos_degrees += self.vel*dt

        # Draw stripe
        self.draw_stripe()

    def draw_stripe(self):
        self.arr.fill( 255 ) # turn all pixels white

        # compute columns of stripe
        stripe_pos_radians = ((180-self.stripe_pos_degrees)*D2R) % (2*np.pi)
        pix_center = stripe_pos_radians/(2*np.pi)*(self.compute_width*8)
        pix_width = 4
        pix_start = int(pix_center-pix_width/2.0)
        pix_stop = pix_start+pix_width

        pix_start = pix_start % (self.compute_width*8)
        pix_stop = pix_stop % (self.compute_width*8)

        if pix_start >= self.arr.shape[1]:
            pix_start = self.arr.shape[1]-1
        if pix_stop >= self.arr.shape[1]:
            pix_stop = self.arr.shape[1]

        # make the stripe pixels black
        self.arr[:,pix_start:pix_stop]=0

        if self.panel_device is not None:
            # send to USB
            try:
                self.panel_device.display_frame(self.arr)
            except:
                sys.stderr.write(
                    'ERROR displaying frame. (Hint: try DISABLE_PANELS=1)\n')
                raise

def rotate_stripe(revs=2,seconds_per_rev=2,fps=50):
    s = StripeClassWorker()
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

def mainloop_server():
    stripe_proxy = StripeClass()
    server = remote_traits.ServerObj(DO_HOSTNAME,DO_PORT)
    server.serve_name('stripe',stripe_proxy)

    stripe_worker = StripeClassWorker()

    def notify_worker_func( obj, name, value ):
        setattr( stripe_worker, name, value)

    stripe_proxy.on_trait_change(notify_worker_func)

    desired_rate = 500.0 #hz
    dt = 1.0/desired_rate

    while 1: # run forever
        server.handleRequests(timeout=dt) # handle network events
        stripe_worker.do_work()

def mainloop():
    stripe_worker = StripeClassWorker()
    desired_rate = 500.0 #hz
    dt = 1.0/desired_rate
    while 1: # run forever
        stripe_worker.do_work()

if __name__=='__main__':
    #rotate_stripe(revs=1,seconds_per_rev=5)
    #rotate_stripe(revs=10,seconds_per_rev=5)
    mainloop()
