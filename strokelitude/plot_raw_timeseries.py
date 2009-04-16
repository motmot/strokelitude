import pkg_resources
import pylab
import numpy as np

import sys
import tables
import motmot.fview_ext_trig.easy_decode as easy_decode

import matplotlib.ticker as mticker

if 1:

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
    chans = r.dtype.fields.keys()
    chans.sort()
    chans.remove('timestamps')

    names = h5.root.ain_wordstream.attrs.channel_names
    if 0:
        Vcc = h5.root.ain_wordstream.attrs.Vcc
    else:
        Vcc=3.3
        print 'Vcc',Vcc
    ADCmax = (2**10)-1
    analog_gain = Vcc/ADCmax

    t0 = r['timestamps'][0]
    N_subplots = len(chans)+2
    ax=None
    for i in range(N_subplots):
        ax = pylab.subplot(N_subplots,1,i+1,sharex=ax)
        if i < len(chans):
            try:
                label = names[int(chans[i])]
            except Exception, err:
                print 'ERROR: ingnoring exception %s'%(err,)
                label = 'channel %s'%chans[i]
            ax.plot(r['timestamps']-t0,r[chans[i]]*analog_gain,
                    label=label)
            ax.set_ylabel('V')
            ax.legend()
        elif i == len(chans):
            ax.plot(stroke_times-t0,stroke_data['right'],label='R')
            ax.set_ylabel('R (degrees)')
            ax.legend()
        elif i == len(chans)+1:
            ax.plot(stroke_times-t0,stroke_data['left'],label='L')
            ax.set_ylabel('L (degrees)')
            ax.legend()
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%s"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%s"))
    ax.set_xlabel('Time (sec)')
    pylab.show()
