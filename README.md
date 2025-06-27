# astro_tools
## Project Description
A compilation of experimental Python tools for handling astrophysical data. The current contents comprise a pipeline for identifying circumbinary exoplanet transits in light curve data from Kepler. More projects will be added as time passes.

## Features
- Cleans and detrends Kepler eclipsing binary (EB) light curves with mean template inference & Wōtan polyfit. Works for EBs with known periods.
- 
For a full explanation of Fourier decomposition & detrend, consult full_documentation.pdf.

## Installation
Best results with Python 3.10 to ensure compatibility of Metal GPU functionality if on an Apple device.
For dependencies, see requirements.txt.

## Use

- TO OBTAIN A DETRENDED LIGHT CURVE FOR ANY MAST KEPLEREBS CATALOG ENTRY:
    - Open prefold_automated.ipynb.
    - Run the first two cells.
    - In the third cell, assign kic to the Kepler Input Catalog ID number of the target.
    - Run all cells to the end of the section 'Detrending'.
    - detrend contains the detrended light curve of the target (as Numpy array).

- TO EVALUATE TENSORFLOW MODEL PERFORMANCE ON REAL DATA:
    - Open prefold_automated.ipynb.
    - Run the first two cells.
    - In the third cell, assign kic to the Kepler Input Catalog ID number of a star with circumbinary planets:
        - KIC 12351927 (Kepler-413)
        - KIC 9632895 (Kepler-453)
        - KIC 4862625 (Kepler-64)
        - KIC 6504534 (Kepler-1661)
        - KIC 6762829 (Kepler-38)
        - KIC 10020423 (Kepler-47)
    - Specify model name in the first cell of 'Evaluation' (only one model 'best_model.keras' at present).
    - Run all cells.
    - Last two cells will display confusion matrix and false/true positive/negative segments.

## File Structure
### 100K_foldless_trainingdata.npz
Simulated training data for tensorflow convolutional neural network (CNN) transit detector.
### a2_train_model_v2.ipynb
Python notebook containing training data synthesis, CNN and model verification protocols.
### keplerebs.villanova.edu
Full light curve dataset of Kepler eclipsing binaries provided by MAST.
### model_training_toolkit.py
Some streamlining functions for tensorflow CNN.
### prefold_automated.ipynb
Python notebook containing full pipeline for dowloading, detrending, & processing Kepler light curves.
### transit_utils.py
Time series packaging functions for non-CNN pipeline steps.

## Contributions

## License
This project is licensed under the terms of the MIT license.

## Acknowledgements
Many thanks to Ming Liu for helping with formulation and debug.

## Future Work
Several elements of the pipeline, CNN & overall project are undergoing extension or debugging. These include:

- CNN layering is being reexamined for efficiency.
- CNN training will be restructured for better handling of edge cases.
- Edge case testing across the board

## References
