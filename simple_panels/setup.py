from setuptools import setup, find_packages

setup(name='motmot.fview.fview2panels',
      namespace_packages = ['motmot','motmot.fview'],

      version='0.1',
      entry_points = {
    'motmot.fview.plugins':'motmot.fview.fview2panels = motmot.fview.fview2panels:FView2Panels_Class',
    },

      packages = find_packages(),
      package_data = {'motmot.fview.fview2panels':['fview2panels.xrc']},
      )
