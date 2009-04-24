import wjUL.UL as UL # apt-get install python-wjul
import time
import numpy as np

BoardNum = 0
chan_left = 0
chan_right = 1
gain =  UL.UNI4VOLTS # works on USB 1208FS

UL.cbAOut(BoardNum, chan_left, gain, 0 )
time.sleep(0.005)
UL.cbAOut(BoardNum, chan_left, gain, 10000 )
time.sleep(0.005)
UL.cbAOut(BoardNum, chan_left, gain, 0 )
time.sleep(0.005)

mean_value = 2
peak_to_peak_amplitude=4
freq_hz = 100#0.5
t_start = time.time()

volts_to_adc_units = 1000.0

while 1:
    t = time.time()-t_start
    left_adc_volts = np.sin( 2*np.pi*t*freq_hz )* peak_to_peak_amplitude*0.5 + mean_value
    left_adc_units = int(left_adc_volts * volts_to_adc_units)
    UL.cbAOut(BoardNum, chan_left, gain, left_adc_units )
    UL.cbAOut(BoardNum, chan_right, gain, left_adc_units )

#left_adc_units = -1000
left_adc_units = 4090

while 1:
    UL.cbAOut(BoardNum, chan_left, gain, left_adc_units )
    UL.cbAOut(BoardNum, chan_right, gain, left_adc_units )
    time.sleep(1)
    print 'output',left_adc_units
    left_adc_units += 1
    #UL.cbAOut(BoardNum, chan_right, gain, left_adc_units )

