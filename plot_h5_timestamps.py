import pkg_resources
import pylab
import numpy as np

import sys
import tables

import matplotlib.ticker as mticker
from optparse import OptionParser
import pytz, datetime, time
pacific = pytz.timezone('US/Pacific')

def format_date(x, pos=None):
    return str(datetime.datetime.fromtimestamp(x,pacific))

def doit(fname,options):
    fname = sys.argv[1]
    h5 = tables.openFile(fname,mode='r')

    stroke_data=h5.root.stroke_data[:]
    print 'stroke_data.dtype',stroke_data.dtype
    frame_numbers = stroke_data['frame']
    stroke_times = stroke_data['trigger_timestamp']
    processing_times = stroke_data['processing_timestamp']
    IFIs = stroke_times[1:]-stroke_times[:-1]

    bad_cond = abs(IFIs) > 1
    idxs = np.nonzero(bad_cond)[0]
    if len(idxs):
        print 'large IFIs:'
        for idx in idxs:
            print 'idx %d, frame %s: %s, IFI %.1f msec'%(
                idx,
                frame_numbers[idx],
                repr(stroke_times[idx]),
                IFIs[idx]*1e3)
            print '  processing time: %s'%repr(processing_times[idx],)

    if 1:
        pylab.plot(frame_numbers[:-1],IFIs*1e3)
        pylab.xlabel('first frame number')
    else:
        pylab.plot(IFIs*1e3)
        pylab.xlabel('index')
    pylab.ylabel('IFI (msec)')
    pylab.show()

def main():
    usage = '%prog [options] FILE'

    parser = OptionParser(usage)

    (options, args) = parser.parse_args()
    fname = args[0]
    doit(fname,options)

if __name__=='__main__':
    main()

