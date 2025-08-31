# astro_tools
## Project Description
A compilation of experimental Python tools for handling astrophysical data. The current contents comprise a pipeline for identifying circumbinary exoplanet transits in light curve data from Kepler. More projects will be added as time passes.

## Features
- Cleans and detrends Kepler eclipsing binary (EB) light curves with convolutional transit mask & Wōtan polyfit. Works for any detached or semi-detached EB with a consistent period.
- Searches processed EB light curves for planetary transits using a convolutional neural network.

## Installation
Best results with Python 3.10 to ensure compatibility of Metal GPU functionality if on an Apple device.
For dependencies, see requirements.txt.

## Use

- TO OBTAIN A DETRENDED LIGHT CURVE FOR ANY MAST KEPLEREBS CATALOG ENTRY:
    - In terminal, input command 'run.sh'. This will open Jupyter notebook prefold_automated.ipynb.
    - Run the first cell.
    - In the first cell of the 'Evaluation' section, set kic_list to the Kepler Input Catalog ID numbers of the desired targets (e.g. kic_list = [kic1, kic2, ...].
    - Run the first two cells of 'Evaluation'.
    - Examine the resulting plot of candidate transits, or extract candidate times using the variable name transit_times.

- TO EVALUATE TENSORFLOW MODEL PERFORMANCE ON KNOWN CIRCUMBINARY PLANETS:
    - In terminal, input command 'run.sh'. This will open Jupyter notebook prefold_automated.ipynb.
    - Run the first cell.
    - In the third cell, assign kic_list to a list of some Kepler Input Catalog ID numbers of stars with circumbinary planets whose transit times are listed in the literature:
        - KIC 12351927 (Kepler-413)
        - KIC 9632895 (Kepler-453)
        - KIC 4862625 (Kepler-64)
        - KIC 5473556 (Kepler-1647)
        - KIC 6504534 (Kepler-1661)
        - KIC 6762829 (Kepler-38)
        - KIC 10020423 (Kepler-47)
    - Run all cells of 'Evaluation'.
    - 

## File Structure
### a2_train_model_v2.ipynb
Python notebook containing training data synthesis, CNN and model verification protocols.
### keplerebs.villanova.edu
Full light curve dataset of Kepler eclipsing binaries provided by MAST.
### prefold_automated.ipynb
Python notebook containing full pipeline for dowloading, detrending, & processing Kepler light curves.
# best_model.keras
Final keras model weights from training in a2_train_model_v2.ipynb.
### requirements.txt
All Python packages needed to run the pipeline.
### run.sh
Shell commands to initalize virtual environment. 
### xgboost_model.json
xgboost decision tree (archive, for comparison purposes).
### rv_search.ipynb
Past project - assembling RV data of nearby stars for JWST (archive, for documentation purposes).
### bp_test.ipynb
Test file for neural net (archive, for documentation purposes).

## License
This project is licensed under the terms of the MIT license.

## Acknowledgements
Many thanks to Ming Liu for helping with formulation and debug.

## Future Work
Several elements of the pipeline & CNN are undergoing extension or debugging. These include:

- CNN layering is being reexamined for efficiency. (done)
- Auto period and transit length determination. (done)
- CNN training will be restructured for better handling of edge cases.
- Edge case testing across the board.
