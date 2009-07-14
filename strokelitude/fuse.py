import sys
from tables import *
from optparse import OptionParser

"""Takes two .h5 files, one with correct analog data
   and the other one with bad analog data and returns
   a single .h5 file with all good data."""

usage = '%prog [options] analog.h5 stroklitude.h5 output.h5'
parser = OptionParser(usage)

(options, args) = parser.parse_args()

if len(args) != 3:
    print 'Please specify two input files and one output file.'
    exit()

analog_fname = args[0]
strokelitude_fname = args[1]
output_fname = args[2]

# Open the two files that are to be fused
analog_h5 = openFile(analog_fname,mode='r')
strokelitude_h5 = openFile(strokelitude_fname,mode='r')

print
print '...---=== Analog file ===---...'
print analog_h5
print '...---=== Strokelitude file ===---...'
print strokelitude_h5

# Output file
fused = openFile(output_fname, mode = "w", title = "Fused file")

# Copy data
analog_h5.root.time_data._f_copy(fused.root)
analog_h5.root.ain_wordstream._f_copy(fused.root)
strokelitude_h5.root.stroke_data._f_copy(fused.root)

print '...---=== Fused file ===---...'
print fused

fused.close()
analog_h5.close()
strokelitude_h5.close()


