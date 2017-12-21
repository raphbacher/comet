# Repositery of the SHADE/COMET method for halo detections

This repositery contains the code for the SHADE approach (based on empirical Benjamini & Hochberg procedure) and the COMET approach for detecting galactic halo on MUSE hyperspectral data

The main application file is shade_main.py
A Shade object is composed of a detection object, a preprocessing object and a postprocessing.

Default parameters are stored in a class Params in parameters.py with specific parameters for preprocessing, detection and postprocessing.

To install:
python setup.py install

To have a step by step tutorial see the notebook Example SHade.ipynb

To test the notebook in a binder environment:
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/raphbacher/comet/master)
