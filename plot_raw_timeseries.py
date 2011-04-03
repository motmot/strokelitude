import pkg_resources
import pylab
import numpy as np

import sys
import tables
import motmot.fview_ext_trig.easy_decode as easy_decode

import matplotlib.ticker as mticker
from optparse import OptionParser
import pytz, datetime, time
pacific = pytz.timezone('US/Pacific')

import scipy.io


def doit(fname,options):
    fname = sys.argv[1]
    h5 = tables.openFile(fname,mode='r')

    stroke_data=h5.root.stroke_data[:]
    stroke_times = stroke_data['trigger_timestamp']

    time_data=h5.root.time_data[:]
    gain,offset,resids = easy_decode.get_gain_offset_resids(
        input=time_data['framestamp'],
        output=time_data['timestamp'])
    top = h5.root.time_data.attrs.top

    wordstream = h5.root.ain_wordstream[:]
    wordstream = wordstream['word'] # extract into normal numpy array

    r=easy_decode.easy_decode(wordstream,gain,offset,top)
    if r is not None:
        chans = r.dtype.fields.keys()
        chans.sort()
        chans.remove('timestamps')
        if 0:
            Vcc = h5.root.ain_wordstream.attrs.Vcc
        print 'Vcc read from file at',Vcc
        else:
            Vcc=3.3
            print 'Vcc',Vcc
        ADCmax = (2**10)-1
        analog_gain = Vcc/ADCmax
    else:
        chans = []
    names = h5.root.ain_wordstream.attrs.channel_names

    if r is not None:
    dt = r['timestamps'][1]-r['timestamps'][0]
    samps_per_sec = 1.0/dt
    adc_duration = n_adc_samples*dt
    print '%d samples at %.1f samples/sec = %.1f seconds'%(n_adc_samples,
                                                           samps_per_sec,
                                                           adc_duration)
        t0 = r['timestamps'][0]
    stroke_times_zero_offset = stroke_times-t0
    if len(stroke_times_zero_offset):
        stroke_data_duration = stroke_times_zero_offset[-1]
        total_duration = max(stroke_data_duration,adc_duration)
    else:
        t0 = 0

    N_subplots = len(chans)+5
    ax=None
    for i in range(N_subplots):
        ax = pylab.subplot(N_subplots,1,i+1,sharex=ax)
        if i < len(chans):
            try:
                label = names[int(chans[i])]
            except Exception, err:
                print 'ERROR: ingnoring exception %s'%(err,)
                label = 'channel %s'%chans[i]
            ax.plot(r['timestamps']-t_offset,r[chans[i]]*analog_gain,
                    label=label)
            ax.set_ylabel('V')
            ax.legend()
        elif i == len(chans):
            if np.all(np.isnan(stroke_data['right'])):
                continue
            ax.set_ylabel('R (degrees)')
            ax.legend()
        elif i == len(chans)+1:
            if np.all(np.isnan(stroke_data['left'])):
                continue
            ax.set_ylabel('L (degrees)')
            ax.legend()
        elif i == len(chans)+2:
            if np.all(np.isnan(stroke_data['left_antenna'])):
                continue
            ax.plot(stroke_times-t0,stroke_data['left_antenna'],label='Lant')
            ax.set_ylabel('L antenna (degrees)')
            ax.legend()
        elif i == len(chans)+3:
            if np.all(np.isnan(stroke_data['right_antenna'])):
                continue
            ax.plot(stroke_times-t0,stroke_data['right_antenna'],label='Rant')
            ax.set_ylabel('R antenna (degrees)')
            ax.legend()
        elif i == len(chans)+4:
            if np.all(np.isnan(stroke_data['head'])):
                continue
            ax.plot(stroke_times-t0,stroke_data['head'],label='H')
            ax.set_ylabel('head (degrees)')
            ax.legend()

            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%s"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%s"))
    ax.set_xlabel('Time (sec)')
    ax.set_xlim((t_plot_start,t_plot_start+total_duration))
    if options.timestamps:
        pylab.gcf().autofmt_xdate()

    pylab.show()

def main():
    usage = '%prog [options] FILE'

    parser = OptionParser(usage)

    parser.add_option("--timestamps", action='store_true',
                      default=False)

    (options, args) = parser.parse_args()
    fname = args[0]
    doit(fname,options)

if __name__=='__main__':
    main()

