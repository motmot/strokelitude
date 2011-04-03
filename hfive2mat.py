import pkg_resources
import pylab
import numpy as np

import sys
import tables
import motmot.fview_ext_trig.easy_decode as easy_decode

from optparse import OptionParser

import matplotlib.ticker as mticker

import scipy.io

parser = OptionParser()
parser.add_option("-p", "--plotmat", action="store_const", const=True, dest="will_plot", help="Plot data using .mat file.")
(options, args) = parser.parse_args()

fname = args[0]
h5 = tables.openFile(fname,mode='r')

stroke_data=h5.root.stroke_data[:]
stroke_times = stroke_data['trigger_timestamp']
print 'repr(stroke_times[0])',repr(stroke_times[0])

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
    if 1:
        Vcc = h5.root.ain_wordstream.attrs.Vcc
        channel_names = h5.root.ain_wordstream.attrs.channel_names
    else:
        Vcc=3.3
    print 'Vcc',Vcc
    ADCmax = (2**10)-1
    analog_gain = Vcc/ADCmax
else:
    chans = []
names = h5.root.ain_wordstream.attrs.channel_names
savedict = {}

if r is not None:
    t0 = r['timestamps'][0]
    savedict = {'ADC_timestamp':r['timestamps']}
else:
    t0 = 0

# Write data to a .mat file
savedict['Stroke_timestamp'] = stroke_times
savedict['Left_wing_angle'] = stroke_data['left']
savedict['Right_wing_angle'] = stroke_data['right'] 
savedict['Left_antenna_angle'] = stroke_data['left_antenna']
savedict['Right_antenna_angle'] = stroke_data['right_antenna']
savedict['Head_angle'] = stroke_data['head']

if chans != []:
    analog_key_list = []
    for i, name in enumerate(names):
        ADC_data = r[chans[i]]*analog_gain
        savedict["ADC"+str(name)] = ADC_data
        analog_key_list.append("ADC"+str(name))
scipy.io.savemat('test.mat',savedict)

def split_dict(dict):
    """ Deletes the dictionary entries which keys are
        __header__, __globals__, etc. Then splits 
        dict into analog and strokelitude dictionaries. """

    keylist = dict.keys()
    analog_keys, strokelitude_keys = [], []
    for key in keylist:
        if key.find('__') != -1:
            del dict[key]
        elif key.find('ADCAIN') != -1:
            analog_keys.append(key)
        elif key.find('timestamp') == -1:
            strokelitude_keys.append(key)
    return analog_keys, strokelitude_keys

if len(args) == 1 and options.will_plot:
    mat = scipy.io.loadmat('test.mat')
    analog_keys, strokelitude_keys = split_dict(mat)
    analog_keys.sort(), strokelitude_keys.sort()
    print analog_keys, strokelitude_keys

    N_analog_subplots = len(analog_keys)
    N_strokelitude_subplots = len(strokelitude_keys)
    N_subplots = N_analog_subplots + N_strokelitude_subplots
    t0 = mat['Stroke_timestamp'][0]
    ax=None
    for i in range(N_strokelitude_subplots):
        ax = pylab.subplot(N_subplots,1,i+1,sharex=ax)
        if np.all(np.isnan(mat[strokelitude_keys[i]])):
            continue
        ax.plot(mat['Stroke_timestamp']-t0,mat[strokelitude_keys[i]],label=strokelitude_keys[i])
        ax.set_ylabel('Angle')
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%s"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%s"))

    for j in range(N_analog_subplots):
        ax = pylab.subplot(N_subplots,1,j + 1 + N_strokelitude_subplots,sharex=ax)
        if np.all(np.isnan(mat[analog_keys[j]])):
            continue
        ax.plot(mat['ADC_timestamp']-t0,mat[analog_keys[j]],label=analog_keys[j])
        ax.set_ylabel('V')
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%s"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%s"))

    ax.set_xlabel('Time (sec)')
    pylab.show()

