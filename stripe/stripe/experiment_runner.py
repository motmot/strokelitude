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

# weird bugs arise in class identity testing if this is not absolute import:
import strokelitude.plugin as strokelitude_plugin_module
import multiprocessing
import multiprocessing.queues
import scipy.interpolate

import ER_data_format

R2D = 180.0/np.pi
D2R = np.pi/180.0

class StateMachine(traits.HasTraits):
    """runs in stimulus control process, communicates with user script via queues"""
    current_state = traits.Str('offline')

    def __init__(self, to_state_machine, from_state_machine,
                 stimulus_state_queue,stimulus_timeseries_queue,
                 display_text_queue):
        super(StateMachine,self).__init__()
        self.to_state_machine = to_state_machine
        self.from_state_machine = from_state_machine

        self.gain = -40.0
        self.offset = 0.0

        self.stripe_pos_degrees = 0.0
        self.last_time = time.time()

        self.current_state_finished_time = None

        self.stimulus_state_queue = stimulus_state_queue
        self.stimulus_timeseries_queue = stimulus_timeseries_queue
        self.display_text_queue = display_text_queue

        self.save_sequence = False
        self.save_sequence_t_start = None
        self.saved_sequence_times = []
        self.saved_sequence_positions = []
        self.saved_sequence_vels = []

    def _current_state_changed(self):
        # see ER_data_format.py
        self.stimulus_state_queue.put( (time.time(),self.current_state) )

    def tick(self,delta_degrees):
        # update stripe position...
        try:
            incoming_message = self.to_state_machine.get_nowait()
        except multiprocessing.queues.Empty:
            pass
        else:
            print 'got message',incoming_message
            if incoming_message.startswith('CL '):
                parts = incoming_message.split()
                tmp, gain, offset, duration_sec, save_sequence = parts
                self.gain = float(gain)
                self.offset = eval(offset) # could instantiate SinusoidalBias class
                duration_sec = float(duration_sec)
                self.save_sequence = bool(int(save_sequence))
                self.current_state = incoming_message
                now = time.time()
                self.current_state_finished_time = now+duration_sec
                print 'switching to closed loop'
                if save_sequence:
                    self.save_sequence_t_start = None
                    self.saved_sequence_times = []
                    self.saved_sequence_positions = []
                    self.saved_sequence_vels = []
                    print 'saving sequence for replay (cleared any old saved sequence)'
                self.display_text_queue.put(
                    'closed loop for %.1f sec (gain=%.1f, offset=%s)'%(
                    duration_sec, self.gain, self.offset))
            elif incoming_message == 'REPLAY':
                self.saved_sequence_times = np.array(self.saved_sequence_times)
                self.saved_sequence_vels = np.array(self.saved_sequence_vels)
                print 'replay times:',self.saved_sequence_times[0],self.saved_sequence_times[-1]
                self.vel_interp = scipy.interpolate.interp1d( self.saved_sequence_times,
                                                              self.saved_sequence_vels )

                # interpolation is slightly tricky here...
                replay_rad = np.unwrap(np.array(self.saved_sequence_positions)*D2R)
                self.pos_interp = scipy.interpolate.interp1d( self.saved_sequence_times,
                                                              replay_rad )

                duration_sec = self.saved_sequence_times[-1]
                now = time.time()
                self.current_state_finished_time = now+duration_sec
                self.save_sequence_t_start = None
                self.current_state = incoming_message
                self.display_text_queue.put(
                    'replaying saved stimulus')
            elif incoming_message.startswith('OLs '):
                parts = incoming_message.split()
                tmp, start_pos_deg, stop_pos_deg, velocity_dps = parts
                start_pos_deg, stop_pos_deg, velocity_dps = map(float,
                                                                (start_pos_deg,
                                                                 stop_pos_deg,
                                                                 velocity_dps))
                self.current_state = incoming_message
                total_deg = stop_pos_deg-start_pos_deg
                duration_sec = total_deg/velocity_dps
                if duration_sec < 0:
                    raise ValueError('impossible combination')
                self.gain = 0
                self.offset = velocity_dps
                self.stripe_pos_degrees = start_pos_deg
                now = time.time()
                self.current_state_finished_time = now+duration_sec
                print 'switching to open loop sweep'
                self.display_text_queue.put(
                    'open loop sweep from %.1f to %.1f at %.1f deg/sec for %.1f sec'%(
                    start_pos_deg, stop_pos_deg, velocity_dps,
                    duration_sec,))
            else:
                raise ValueError('Unknown message: %s'%incoming_message)

        if self.current_state != 'offline':
            now = time.time()
            if now >= self.current_state_finished_time:
                print 'switching to offline because current state ended'
                outgoing_message = 'done '+self.current_state
                self.from_state_machine.put(outgoing_message)
                self.current_state = 'offline'
                self.save_sequence = False

        now = time.time()

        if self.current_state == 'REPLAY':
            if self.save_sequence_t_start is None:
                self.save_sequence_t_start = now
            seq_t = now-self.save_sequence_t_start
            vel_dps = self.vel_interp(seq_t)
            pos_rad_wrapped = self.pos_interp(seq_t)
            self.stripe_pos_degrees = (pos_rad_wrapped*R2D+180)%360-180
        else:

            if isinstance(self.offset,SinusoidalBias):
                offset = self.offset.update(now)
            else:
                offset = self.offset

            # update self.vel every frame so that changes in gain and offset noticed
            vel_dps = delta_degrees*self.gain + offset

            # Compute stripe position.
            dt = now-self.last_time
            self.last_time = now

            self.stripe_pos_degrees += vel_dps*dt
            # put in range -180 <= angle < 180
            self.stripe_pos_degrees = (self.stripe_pos_degrees+180)%360-180

            if self.save_sequence:
                if self.save_sequence_t_start is None:
                    self.save_sequence_t_start = now
                seq_t = now-self.save_sequence_t_start
                self.saved_sequence_times.append( seq_t )
                self.saved_sequence_positions.append( self.stripe_pos_degrees )
                self.saved_sequence_vels.append( vel_dps )

        return self.stripe_pos_degrees, vel_dps

class SinusoidalBias:
    def __init__(self, freq_hz=None, amplitude=None):
        self.freq_hz = freq_hz
        self.amplitude = amplitude
        self.start_t = None
    def __repr__(self):
        return 'SinusoidalBias(freq_hz=%s,amplitude=%s)'%(repr(self.freq_hz),
                                                          repr(self.amplitude))
    def update(self,now=None):
        if self.start_t is None:
            self.start_t = now
        t = now-self.start_t
        phase = t*(2*np.pi*self.freq_hz)
        val = self.amplitude*np.sin(phase)
        return val

class StripeControlRemoteProcess:
    """Runs in the same process as the user's experiment script"""
    def __init__(self,to_state_machine,from_state_machine,display_text_queue):
        self.to_state_machine = to_state_machine
        self.from_state_machine = from_state_machine
        self.display_text_queue = display_text_queue
    def set_progress(self, n_done, total_repititions ):
        print '%d (of %d) done'
    def show_string( self, msg ):
        self.display_text_queue.put(msg)
    def closed_loop( self, gain=-900.0, offset=0.0, duration_sec=30, save_sequence=False ):
        to_msg = 'CL %s %s %s %s'%(repr(gain),repr(offset),repr(duration_sec), repr(int(save_sequence)))
        self._send(to_msg)
    def _send(self,to_msg):
        print 'sending message to state machine',to_msg
        self.to_state_machine.put(to_msg)
        expected_from_msg = 'done '+to_msg
        print 'awaiting message from state machine'
        from_msg = self.from_state_machine.get()
        if from_msg != expected_from_msg:
            print 'from_msg',repr(from_msg)
            print 'expected_from_msg', repr(expected_from_msg)
            raise RuntimeError('did not get expected message')
    def open_loop_sweep( self,
                         start_pos_deg = 180,
                         stop_pos_deg = -180,
                         velocity_dps = 100 ):
        to_msg = 'OLs %s %s %s'%(repr(start_pos_deg),
                                 repr(stop_pos_deg),
                                 repr(velocity_dps))
        self._send(to_msg)
    def replay_sequence( self ):
        to_msg = 'REPLAY'
        self._send(to_msg)

def stripe_control_runner(experiment_file,
                          to_state_machine,
                          from_state_machine,
                          display_text_queue):
    # this runs in a separate process
    print 'running experiment',experiment_file
    namespace = dict( stripe_control=StripeControlRemoteProcess(to_state_machine,
                                                                from_state_machine,
                                                                display_text_queue),
                      SinusoidalBias=SinusoidalBias)
    execfile(experiment_file,namespace)

class ExperimentRunnerPluginInfo(strokelitude_plugin_module.PluginInfoBase):
    """called by strokelitude.plugin"""
    name = traits.Str('Stripe Experiment Runner')

    def get_hastraits_class(self):
        return StripeClass, StripeClassWorker
    def get_worker_table_descriptions(self):
        return {
            'stimulus_state_queue':ER_data_format.StateDescription,
            'stimulus_timeseries_queue':ER_data_format.TimeseriesDescription}

class StripeClass(remote_traits.MaybeRemoteHasTraits):
    """base class of worker subclass, and also runs in GUI process"""
    experiment_file = traits.File()
    start_experiment = traits.Button(label='start experiment')

    traits_view = View( Group( (
        Item(name='start_experiment',show_label=False),
        Item(name='experiment_file'),
        )),
                        )

class StripeClassWorker(StripeClass):
    """runs in process that updates panels"""
    def __init__(self,
                 stimulus_state_queue=None,
                 stimulus_timeseries_queue=None,
                 display_text_queue=None,
                 ):
        super(StripeClassWorker,self).__init__()
        if simple_panels is not None:
            self.panel_device = simple_panels.DumpframeDevice()
        else:
            self.panel_device = None

        self.panel_height=4
        self.panel_width=11 # number of actual panels
        self.compute_width=12 # number of panels to complete a full circles
        self.zero_column = 44 # column number of straight ahead
        self.stripe_pos_degrees = 0.0
        self.vel_dps = 0.0
        self.last_frame = 0
        self.receive_timestamp = 0.0
        self.trigger_timestamp = 0
        self.arr = np.zeros( (self.panel_height*8, self.panel_width*8),
                              dtype=np.uint8)
        self.vel = 0.0
        self.last_diff_degrees = 0.0
        self.incoming_data_queue = Queue.Queue() # temporary until set by host

        self.to_state_machine = multiprocessing.Queue()
        self.from_state_machine = multiprocessing.Queue()
        self.stimulus_state_queue = stimulus_state_queue
        self.stimulus_timeseries_queue = stimulus_timeseries_queue
        self.display_text_queue = display_text_queue
        self.state_machine = StateMachine(
            self.to_state_machine,
            self.from_state_machine,
            stimulus_state_queue=self.stimulus_state_queue,
            stimulus_timeseries_queue=self.stimulus_timeseries_queue,
            display_text_queue=self.display_text_queue )

    def _start_experiment_fired(self):
        if not self.experiment_file:
            raise ValueError('no experiment file specified')

        ## if not self.is_offline:
        ##     raise RuntimeError('already running experiment!')

        self.sci = multiprocessing.Process(target=stripe_control_runner,
                                           args=(self.experiment_file,
                                                 self.to_state_machine,
                                                 self.from_state_machine,
                                                 self.display_text_queue))

        # notify GUI that we are running experiment...
        ## self.is_offline = False
        self.sci.daemon = True # don't keep parent open
        self.sci.start()

    def _stop_experiment_fired(self):
        self.to_state_machine.put('abort')

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
            (framenumber,
             left_angle_degrees, right_angle_degrees,
             trigger_timestamp) = last_data
            self.last_frame = framenumber
            self.trigger_timestamp = trigger_timestamp
            self.receive_timestamp = time.time()
            if not (np.isnan(left_angle_degrees) or
                    np.isnan(right_angle_degrees)):
                # (opposite signs already from angle measurement)
                diff_degrees = right_angle_degrees - left_angle_degrees
                self.last_diff_degrees = diff_degrees

        self.stripe_pos_degrees, self.vel_dps = \
                                 self.state_machine.tick(self.last_diff_degrees)

        # Draw stripe
        self.draw_stripe()

    def draw_stripe(self):
        self.arr.fill( 255 ) # turn all pixels white

        # compute columns of stripe
        stripe_pos_radians = self.stripe_pos_degrees*D2R
        pix_center = self.zero_column-(stripe_pos_radians/(2*np.pi)*(self.compute_width*8))
        pix_width = 4
        pix_start = int(pix_center-pix_width/2.0)
        pix_stop = pix_start+pix_width

        pix_start = pix_start % (self.compute_width*8)
        pix_stop = pix_stop % (self.compute_width*8)

        # XXX there is some funkiness here
        if pix_start >= self.arr.shape[1]:
            pix_start = self.arr.shape[1]-1
        if pix_stop >= self.arr.shape[1]:
            pix_stop = self.arr.shape[1]

        # make the stripe pixels black
        self.arr[:,pix_start:pix_stop]=0

        start_display_timestamp = time.time()

        if self.panel_device is not None:
            # send to USB
            try:
                self.panel_device.display_frame(self.arr)
            except:
                sys.stderr.write(
                    'ERROR displaying frame. (Hint: try DISABLE_PANELS=1)\n')
                raise

        stop_display_timestamp = time.time()
        # see ER_data_format.py
        self.stimulus_timeseries_queue.put( (self.last_frame,
                                             self.trigger_timestamp,
                                             self.receive_timestamp,
                                             start_display_timestamp,
                                             stop_display_timestamp,
                                             self.stripe_pos_degrees,
                                             self.vel_dps) )

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
    if 1:
        stripe_worker.configure_traits()
    else:
        print 'running forever...'
        desired_rate = 500.0 #hz
        dt = 1.0/desired_rate
        while 1: # run forever
            stripe_worker.do_work()

if __name__=='__main__':
    #rotate_stripe(revs=1,seconds_per_rev=5)
    #rotate_stripe(revs=10,seconds_per_rev=5)
    mainloop()
