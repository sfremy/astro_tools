# astro_tools
## Project Description
A compilation of experimental Python tools for handling astrophysical data. The current contents comprise a pipeline for identifying circumbinary exoplanet transits in light curve data from Kepler. More projects will be added as time passes.

## Features
- Cleans, detrends and searches for exoplanet transit signals using Fourier decomposition. Can handle eclipsing binary targets with known periods.
- Scans processed light curves and flags systems with signs of exoplanet transits.
For a full explanation of Fourier decomposition & detrend, consult full_documentation.pdf.

## Installation
Best results with Python 3.10 to ensure compatibility of Metal GPU functionality if on an Apple device.
For dependencies, see requirements.txt.

## Use
Two use cases are currently possible:

- Provide the Kepler Input Catalog number of a target star into prefold_automated first cell. Run the first three cells, which comprise the complete pipeline.
- Design a set of input filters into the fourth cell to run a restricted survey of the keplerebs catalog.

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
Time series handling functions for non-CNN pipeline steps.

## Contributions

## License
This project is licensed under the terms of the MIT license.

## Acknowledgements
Many thanks to Ming Liu for helping with formulation and debug.

## Future Work
Several elements of the pipeline, CNN & overall project are undergoing extension or debugging. These include:

- Eclipse mask filling behaves incorrectly in some situations. Anomalies are currently being diagnosed.
- CNN layering is being reexamined for efficiency.
- CNN training will be restructured for better handling of edge cases.
- Edge case testing across the board

## References
