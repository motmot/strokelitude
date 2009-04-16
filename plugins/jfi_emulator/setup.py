from setuptools import setup, find_packages

setup(name='jfi_emulator',
      version='0.1',
      entry_points = {
    'strokelitude.plugins':[
    'JFIEmulatorPluginInfo = jfi_emulator.emulator:JFIEmulatorPluginInfo',
    ],
    },
      packages = find_packages(),
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      )
