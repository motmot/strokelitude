import tables
import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
h5=tables.openFile(filename,mode='r')
stim=h5.root.stimulus_timeseries_queue[:]
stroke_data =h5.root.stroke_data[:]

fig=plt.figure()
ax1=fig.add_subplot(2,1,1)

latencies_sec = {}
for name in ['receive_timestamp',
             'start_display_timestamp',
             'stop_display_timestamp']:

    latencies_sec[name] = stim[name]-stim['trigger_timestamp']

    ax1.plot(stim['trigger_timestamp'],
             1000.0*latencies_sec[name],
             '.',
             label=name)

latencies_sec['image_analysis'] = (stroke_data['processing_timestamp']-
                                   stroke_data['trigger_timestamp'])
ax1.plot( stroke_data['trigger_timestamp'],
         1000.0*latencies_sec['image_analysis'], '.',
         label='image analysis')

ax1.set_xlabel('time (sec)')
ax1.set_ylabel('latency (msec)')
ax1.legend()

ax2=fig.add_subplot(2,1,2)
bins = np.linspace(0,100,30)
for name,results in latencies_sec.iteritems():
    ax2.hist(results*1000.0,bins=bins,alpha=0.5,label=name)
ax2.set_xlabel('latency (msec)')
ax2.legend()
plt.show()
