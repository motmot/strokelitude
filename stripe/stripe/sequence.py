import strokelitude.plugin
import remote_traits
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor
import Queue

global state
state = 'unknown'

class SequencePluginInfo(strokelitude.plugin.PluginBase):
    def get_name(self):
        return 'Sequence experiment'
    def get_hastraits_class(self):
        return SequenceClass, SequenceClassWorker

class SequenceClass(remote_traits.MaybeRemoteHasTraits):
    start = traits.Button(label='start')
    stop = traits.Button(label='stop')

    traits_view = View( HGroup( ( Item( 'start', show_label = False ),
                                  Item( 'stop', show_label = False ),
                                 )),
                        )

class SequenceClassWorker(SequenceClass):
    def _start_fired(self):
        print 'starting'
    def _stop_fired(self):
        print 'stopping'

    def __init__(self):
        self.incoming_data_queue = Queue.Queue()
        self._last_state = None

    def set_incoming_queue(self,data_queue):
        self.incoming_data_queue = data_queue

    def do_work( self ):
        """This gets called frequently (e.g. 100 Hz)"""
        global state

        ## # Get any available incoming data. Ignore all but most recent.
        last_data = None
        while 1:
            try:
                last_data = self.incoming_data_queue.get_nowait()
            except Queue.Empty, err:
                break

        if state != self._last_state:
            print state
            self._last_state = state

def closed_loop(dur_sec):
    global state
    state = 'closed loop'
    time.sleep(dur_sec)

def open_loop(dur_sec):
    global state
    state = 'open loop'
    time.sleep(dur_sec)

def sequence():
    global state
    print 'starting sequence'
    for i in range(5):
        closed_loop(10.0)
        open_loop(5.0)
        closed_loop(10.0)
        open_loop(5.0)

    state = 'unknown'
