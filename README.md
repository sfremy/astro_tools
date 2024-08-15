# astro_tools
A compilation of Python tools for handling data analysis problems in astrophysics. 
## Contents
### cb_planets
Contains a complete pipeline which searches Kepler/TESS eclipsing binary light curves for planetary transits using a convolutional neural network. Tookit includes the following:
- Kepler/TESS data downloader. Searches published datasets for catalog ID matches, then filters based on user specifications.
- By-segment time series detrending tool. Uses variable-length Fourier transform to eliminate flux variations caused by stellar activity in downloaded Kepler/TESS data.
- Masking tool for eclipsing binaries. Obtains parameters of stellar eclipses and smooths them out.
- Artificial data generator for convolutional neural network.
- Custom-trained convolutional neural network.
- Test architecture for convolutional neural network.
### Additional projects will be added here.

## Installation
(WIP)

## How to Use
(WIP)

## Credits
Many thanks to Ming Liu for helping with formulation and debug.
