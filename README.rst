**********************************************************************************
:mod:`strokelitude` - realtime image analysis for tethered flight fly experiments.
**********************************************************************************

.. module:: strokelitude
  :synopsis: realtime image analysis for tethered flight fly experiments
.. index::
  module: strokelitude
  single: strokelitude

Strokelitude (wingSTROKE ampLITUDE) is an `fview
<http://code.astraw.com/projects/motmot/fview.html>`_ plugin for
realtime image analysis for tethered flight fly experiments. It was
originally developed in the `Dickinson lab <http://flyranch.org>`_ at
Caltech, and is now maintained by the `Straw lab
<http://strawlab.org>`_. One goal is to emulate the optical wingstroke
analyzer first developed by Karl Götz in Tübingen.

The software has been used as a key method in the following papers:

 * Maimon, G., Straw, A.D., and Dickinson, M.H. (2010). Active flight
   increases the gain of visual motion processing in
   Drosophila. Nature Neuroscience. `PDF
   <http://code.astraw.com/MaimonStrawDickinson_2010.pdf>`_ and
   `Supplementary video
   <http://www.nature.com/neuro/journal/v13/n3/extref/nn.2492-S2.mov>`_.

 * Mamiya, A., Straw, A.D., Tómasson, E., and Dickinson, M.H. (2011)
   Active and passive antennal movements during visually-guided
   steering in flying Drosophila. Journal of Neuroscience.

Initial development was done in the laboratory of Michael
Dickinson. Development was led by Andrew Straw. Egill Tómasson
develped the antennae and head trackers. Gaby Maimon and Akira Mamiya
provided continuous suggestions and beta testing. Additional input
came from Peter Polidoro, Will Dickson, and Allan Wong.

This is open source software, released under the BSD license. The
source code is available at
https://github.com/motmot/strokelitude. Discussion happens at the
`motmot email list
<http://code.astraw.com/cgi-bin/mailman/listinfo/motmot>`_.

The following plugin is available:

 * :mod:`jfi_emulator` - output amplitude of wingstroke envelope as
   analog voltage
