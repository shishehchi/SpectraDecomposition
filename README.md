# Spectra Decomposition
Dissection of composite spectra of galaxies into individual components using deep supervised learning techniques.

[![DOI](https://zenodo.org/badge/731416625.svg)](https://zenodo.org/doi/10.5281/zenodo.10398092)

# Summary

Galaxies are vast and intricate systems that host diverse regions with different stellar populations and characteristics. With the emergence of IFU surveys such as MaNGA, astronomers can now study these regions in more detail since such surveys offer more precise measurements of critical physical parameters. However, IFU surveys mainly target specific galaxies in the local Universe, which may limit their generalizability. In contrast, single-slit spectroscopic surveys like SDSS capture integrated light from various populations, providing an average physical property of the galaxy. For instance, extracting the SFR or metallicity of galaxies from integrated spectra can be challenging due to the mixture of contributions from different region

In order to overcome the challenges of understanding the complex nature of galaxies and their underlying physical processes, we have developed a method to decompose SDSS spectra into two distinct components: Active Galactic Nuclei (AGN) and Star-Forming (SF) populations. To demonstrate this approach, we have used MaNGA spectra as a training set. Our method involves applying deep supervised learning techniques, trained on IFU surveys, to decompose a single composite spectrum into two separate spectra and to determine the contribution of each population.


# This Repository

To test our trained model, please download the [classifier](https://drive.google.com/file/d/1WdXHMr5N4mIm4e435TnUKLANLceMVxMw/view?usp=sharing) and [regressor](https://drive.google.com/file/d/1F-JMK59GhILmNt_PkEErw1BF63Nd_3yZ/view?usp=sharing) model weights and save them in the "weights" folder. Download the [wavelength](https://drive.google.com/file/d/1Amk7qBBVBZDxKTy3KZar4OglYndmnuMg/view?usp=sharing) data and save them in the "data" folder. Then use the piplene.ipynb file to test the model on your data. Input data to the model must be prepared considering the following requirements:

Provide one file that contains the input spectra as follows. Save the file in the data folder.

1- Input spectra must be resampled to 2A wavelength bins between 3700 A and 8114 A. Therefore, each input spectra must have 2208 elements.
  
2- All the major emission lines (Ha, Hb, O3, N2 and S2) need to have signal-to-noise ratio values greater than 8.
  
Provide one file that contains the corresponding weight of each spectra. Save it in the data folder.

Alternatively, you can test the code with our data sample that you can download [here](https://drive.google.com/file/d/1NMtspP9GkSTI1vZDHN1uvqzq52UX175G/view?usp=sharing). Please unzip the file and add the two .npy files to the data folder.

We normalize each MaNGA spectra by first subtracting the median value of the spectra between data points 4000 A and 7000 A, which covers the four emission lines required for placement on a BPT diagram. Then, by dividing it to the fiber mass and the standrad deviation of the spectra between data points 4000 A and 7000 A.
  


# The deep decomposition model
Our Deep Decomposition Model (DDM) consists of a model with five regressors, each with its own loss function. A detailed information of the structure of the model is below:

<img src="images/model_architecture.png" width="100%">
