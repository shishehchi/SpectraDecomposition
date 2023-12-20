# Spectra Decomposition
Dissection of composite spectra of galaxies into individual components using deep supervised learning techniques.

[![DOI](https://zenodo.org/badge/731416625.svg)](https://zenodo.org/doi/10.5281/zenodo.10398092)

# Summary

Traditional single-slit spectroscopy yields a single composite spectrum that encapsulates the intricate internal structures and activities within a galaxy. These spectra serve as the foundation for estimating crucial galaxy parameters, but their accuracy can be compromised by various contributing sources. In contrast, Integral Field Unit (IFU) surveys minimize the effects of mixing different sources by examining spatially localized regions within a galaxy. Nevertheless, they often exhibit limitations concerning the diversity and number of galaxies covered, typically focusing on nearby galactic systems.

In this study, we introduce a novel model informed by insights from the MaNGA IFU survey. This model allows us to deconstruct galaxy spectra, including those sourced from the Sloan Digital Sky Survey (SDSS), into their SF and AGN constituents. Application of our model to these survey datasets yields two distinct spectra—one for SF and another for AGN components—while maintaining empirical flux conservation across wavelength bins.

# This Repository

To test our trained model, please download the [classifier](https://drive.google.com/file/d/1WdXHMr5N4mIm4e435TnUKLANLceMVxMw/view?usp=sharing) and [regressor](https://drive.google.com/file/d/1F-JMK59GhILmNt_PkEErw1BF63Nd_3yZ/view?usp=sharing) model weights and save them in the "weights" folder. Download the [wavelength](https://drive.google.com/file/d/1Amk7qBBVBZDxKTy3KZar4OglYndmnuMg/view?usp=sharing) data and save them in the "data" folder. Then use the piplene.ipynb file to test the model on your data. Input data to the model must be prepared considering the following requirements:

Provide one file that contains the input spectra as follows. Save the file as data/input.npy

1- Input spectra must be resampled to 2A wavelength bins between 3700 A and 8114 A. Therefore, each input spectra must have 2208 elements.
  
2- All the major emission lines (Ha, Hb, O3, N2 and S2) need to have signal-to-noise ratio values greater than 8.
  
Provide one file that contains the corresponding weight of each spectra. Save it as data/weights.npy

We normalize each MaNGA spectra by first subtracting the median value of the spectra between data points 4000 A and 7000 A, which covers the four emission lines required for placement on a BPT diagram. Then, by dividing it to the fiber mass and the standrad deviation of the spectra between data points 4000 A and 7000 A.
  


# The deep decomposition model
Our Deep Decomposition Model (DDM) consists of a model with five regressors, each with its own loss function. A detailed information of the structure of the model is below:

<img src="images/model_architecture.png" width="100%">
