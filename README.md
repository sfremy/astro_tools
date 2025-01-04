# astro_tools
## Prject Description
A compilation of experimental Python tools for handling astrophysical data. The current contents comprise a pipeline for identifying circumbinary exoplanet transits in light curve data from Kepler. More projects will be added as time passes.

## Features

## Installation
Best results with Python 3.10 to ensure compatibility of Metal GPU functionality if on an Apple device. 

## Use
Provide the Kepler Input Catalog number of a target star(s).
Use provided notebooks for data visualisation & analysis.

## File Structure
### 100K_foldless_trainingdata.npz
Simulated training data for tensorflow convolutional neural network (CNN) transit detector.
### a2_train_model_v2.ipynb
Python notebook containing training data synthesis, CNN and model verification protocols. (Under review)
### keplerebs.villanova.edu
Full light curve dataset of Kepler eclipsing binaries provided by MAST.
### model_training_toolkit.py
Some streamlining functions for tensorflow CNN. (Under review)
### prefold_automated.ipynb
Python notebook containing full pipeline for dowloading, detrending, & processing Kepler light curves. At present, the pipeline has the following capabilities:

- Loading & cleaning Kepler/TESS light curves
- Detecting & masking stellar eclipses
- Fourier detrending Kepler/TESS light curves
- Identifying possible planetary transits using CNN
  
### transit_utils.py
Some streamlining functions for non-model pipeline elements. (Under review)

## Contributions

## License

## Acknowledgements
Many thanks to Ming Liu for helping with formulation and debug.

## Future Work
Several elements of the pipeline, CNN & overall project are undergoing extension or debugging. These include:

- Eclipse mask filling behaves incorrectly in some situations. Anomalies are currently being diagnosed.
- CNN layering is being reexamined for efficiency.
- CNN training will be restructured for better handling of edge cases.
- Edge case testing across the board

## References
