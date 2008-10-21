from setuptools import setup, find_packages

setup(name='motmot.fview.strokelitude',
      namespace_packages = ['motmot','motmot.fview'],

      version='0.1',
      entry_points = {
    'motmot.fview.plugins':'motmot.fview.strokelitude = motmot.fview.strokelitude:StrokelitudeClass',
    },

      packages = find_packages(),
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      zip_safe=True,
      package_data = {'motmot.fview.strokelitude':['fview_strokelitude.xrc']},
      )
