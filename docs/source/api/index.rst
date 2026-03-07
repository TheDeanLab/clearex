API Reference
=============

The API reference is generated from docstrings in ``src/clearex``.

Core Runtime
------------

.. autosummary::
   :toctree: generated

   clearex.main
   clearex.workflow

I/O, Experiment, and Provenance
-------------------------------

.. autosummary::
   :toctree: generated

   clearex.io.cli
   clearex.io.dask
   clearex.io.experiment
   clearex.io.log
   clearex.io.provenance
   clearex.io.read

Analysis Modules
----------------

.. autosummary::
   :toctree: generated

   clearex.deconvolution.petakit
   clearex.detect.particles
   clearex.detect.pipeline
   clearex.filter.filters
   clearex.filter.kernels
   clearex.fit.points
   clearex.preprocess.pad
   clearex.preprocess.scale
   clearex.registration
   clearex.registration.common
   clearex.registration.linear
   clearex.registration.nonlinear
   clearex.registration.tre
   clearex.segmentation.otsu
   clearex.segmentation.pointsource
   clearex.segmentation.regions
   clearex.segmentation.tubes
   clearex.stats.intensity

Visualization and GUI
---------------------

.. autosummary::
   :toctree: generated

   clearex.gui.app
   clearex.visualization.napari
   clearex.visualization.pipeline

Utilities
---------

.. autosummary::
   :toctree: generated

   clearex.context.environment
   clearex.file_operations.dask
   clearex.file_operations.tools
   clearex.plot.data
   clearex.plot.images
