import pylab
import numpy as np

import sys
import tables
import fview_ext_trig.easy_decode as easy_decode

if 1:

    fname = sys.argv[1]
    h5 = tables.openFile(fname,mode='r')

    stroke_data=h5.root.stroke_data[:]
    stroke_times = stroke_data['trigger_timestamp']
    stroke_diff = stroke_data['right']-stroke_data['left']

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

    t0 = r['timestamps'][0]
    N_subplots = len(chans)+3
    ax=None
    for i in range(N_subplots):
        ax = pylab.subplot(N_subplots,1,i+1,sharex=ax)
        if i < len(chans):
            ax.plot(r['timestamps']-t0,r[chans[i]],label=names[int(chans[i])])
            ax.set_ylabel('(ADC units)')
            ax.legend()
        elif i == len(chans):
            ax.plot(stroke_times-t0,stroke_diff,label='R-L')
            ax.set_ylabel('R-L (degrees)')
            ax.legend()
        elif i == len(chans)+1:
            ax.plot(stroke_times-t0,stroke_data['right'],label='R')
            ax.set_ylabel('R (degrees)')
            ax.legend()
        elif i == len(chans)+2:
            ax.plot(stroke_times-t0,stroke_data['left'],label='L')
            ax.set_ylabel('L (degrees)')
            ax.legend()
    ax.set_xlabel('Time (sec)')
    pylab.show()
