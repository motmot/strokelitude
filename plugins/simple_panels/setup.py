from setuptools import setup, find_packages
import os, sys

libfname='libdumpframe1.so' # XXX change on Windows and Mac
fname =os.path.join('simple_panels',libfname)
if not os.path.exists(fname):
    # XXX should only do this check during build phase
    sys.stderr.write('Error: could not find %s. Run scons. Quitting.\n'%fname)
    sys.exit(1)

setup(name='simple_panels',
      version='0.1',
      packages = find_packages(),
      package_data={'simple_panels':libfname},
      )
