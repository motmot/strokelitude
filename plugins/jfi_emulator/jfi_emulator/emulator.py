# weird bugs arise in class identity testing if this is not absolute import:
import strokelitude.plugin as strokelitude_plugin_module
import remote_traits
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor
import wjUL.UL as UL # apt-get install python-wjul
import Queue
import numpy as np

class JFIEmulatorPluginInfo(strokelitude_plugin_module.PluginInfoBase):
    """called by strokelitude.plugin"""
    name = traits.Str('JFI Emulator')

    def get_hastraits_class(self):
        return JFIEmulatorClass, JFIEmulatorClassWorker
    def get_worker_table_descriptions(self):
        return {}

class JFIEmulatorClass(remote_traits.MaybeRemoteHasTraits):
    """base class of worker subclass, and also runs in GUI process"""
    max_voltage = traits.Float(4.0)
    volts_to_adc_units = traits.Float(1000) # conversion factor

    BoardNum = traits.Int(0)
    chan_left = traits.Int(0)
    chan_right = traits.Int(1)
    min_angle = traits.Float(-45) # in same units as sent by strokelitude
    max_angle = traits.Float(90) # in same units as sent by strokelitude
    gain =  UL.UNI4VOLTS # works on USB 1208FS

    traits_view = View( Group( (
        ## Item(name='max_voltage'),
        ## Item(name='volts_to_adc_units'),
        )),
                        )
class JFIEmulatorClassWorker(JFIEmulatorClass):
    """runs in process that updates panels"""

    angle_gain = traits.Property(depends_on=['max_voltage','min_angle','max_angle'])
    angle_offset = traits.Property(depends_on=['angle_gain','min_angle'])

    def __init__(self,
                 ## stimulus_state_queue=None,
                 ## stimulus_timeseries_queue=None,
                 display_text_queue=None,
                 ):
        super(JFIEmulatorClass,self).__init__()
        self.incoming_data_queue = Queue.Queue() # temporary until set by host

    @traits.cached_property
    def _get_angle_gain(self):
        return self.max_voltage / (self.max_angle - self.min_angle)

    @traits.cached_property
    def _get_angle_offset(self):
        return -self.angle_gain * self.min_angle

    def _stop_experiment_fired(self):
        pass

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

        # Update voltages if new data arrived.
        if last_data is not None:
            (framenumber,
             left_angle_degrees, right_angle_degrees,
             trigger_timestamp) = last_data

            if not np.isnan(left_angle_degrees):
                left_adc_volts = left_angle_degrees * self.angle_gain + self.angle_offset
                if left_adc_volts < 0:
                    left_adc_volts = 0
                if left_adc_volts > self.max_voltage:
                    left_adc_volts = self.max_voltage
                left_adc_units = int(left_adc_volts * self.volts_to_adc_units)
                UL.cbAOut(self.BoardNum, self.chan_left, self.gain, left_adc_units )

            if not np.isnan(right_angle_degrees):
                right_adc_volts = right_angle_degrees * self.angle_gain + self.angle_offset
                if right_adc_volts < 0:
                    right_adc_volts = 0
                if right_adc_volts > self.max_voltage:
                    right_adc_volts = self.max_voltage
                right_adc_units = int(right_adc_volts * self.volts_to_adc_units)
                UL.cbAOut(self.BoardNum, self.chan_right, self.gain, right_adc_units )
