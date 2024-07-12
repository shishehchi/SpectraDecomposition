# Spectra Decomposition
Revisiting AGN Placement on the BPT Diagram: A Spectral Decomposition Approach
[![DOI](https://zenodo.org/badge/731416625.svg)](https://zenodo.org/doi/10.5281/zenodo.10398092)


# Abstract 
Traditional single-slit spectroscopy provides a single galaxy spectrum, forming the basis for crucial parameter estimation. However, its accuracy can be compromised by various sources of contamination, such as the prominent Hα emission line originating from both Star-Forming (SF) regions and Active Galactic Nuclei (AGN). The potential to dissect a spectrum into its SF and AGN constituents holds the promise of significantly enhancing precision in parameter estimates. In contrast, Integral Field Unit (IFU) surveys present a solution to minimize contamination. These surveys examine spatially localized regions within galaxies, reducing the impact of mixed sources. Although resulting spectra cover smaller regions than single-slit spectroscopy, they can still encompass a blend of heterogeneous sources. Our study introduces an innovative model informed by insights from the MaNGA IFU survey. This model enables the decomposition of galaxy spectra, including those from the Sloan Digital Sky Survey (SDSS), into SF and AGN components. Applying our model to these survey datasets produces two distinct spectra, one for SF and another for AGN components while conserving flux across wavelength bins. Remarkably, when these decomposed spectra are visualized on a BPT diagram, interesting patterns emerge. There is a significant shift in the placement of the AGN decomposed spectra, as well as the emergence of two distinct clusters in the LINER and Seyfert regions. This shift highlights the pivotal role of SF ’contamination’ in influencing the positioning of AGN spectra within the BPT diagram.


# Summary
Galaxies are intricate systems that necessitate systematic classification to facilitate thorough analysis. These classifications include distinguishing between star-forming activities and Active Galactic Nuclei (AGN) phenomena, which are potent energy sources located at the centers of galaxies. AGN galaxies are further sub-divided into categories such as Seyfert and LINER galaxies. In this paper we use the terminology ’SF’ to refer to star-forming regions and ’NonSF’ for non-starforming regions, which may also include true AGNs. 
Galactic research utilizes diverse data acquisition methodologies, such as photometric data collection and integrated single-slit spectroscopy. Notably, the Sloan Digital Sky Survey (SDSS) employs a spectrograph to capture galactic spectra. These spectra may not encompass the entire galaxy, highlighting the importance of the covering fraction. The spectrum that emerges is a composite of emissions from various parts of the galaxy. These complex optical spectra can be dissected to isolate contributions from SF and NonSF activities. Within the framework of the superposition principle, emissions from NonSF and SF regions can be considered as distinct waveforms. This principle asserts that any resultant waveform at a given point, created by the overlap of multiple waves, is simply the algebraic summation of these individual waves. Consequently, the observed emission from a galaxy often represents a blend of contributions from both NonSF and SF regions. While SDSS, an example of a single-slit spectroscopic survey, has significantly contributed to our understanding of galaxy formation and evolution, it is limited by averaging the complex internal structures of target galaxies. To address the limitations of single-fibre spectroscopic surveys like SDSS in capturing the complex internal structure of galaxies, IFU surveys have emerged as a pivotal tool. IFU surveys capture data across smaller twodimensional fields, providing a more nuanced view of the internal structures of galaxies. This approach complements the broader insights offered by surveys like SDSS. A widely utilized IFU dataset is the Mapping Nearby Galaxies at Apache Point Observatory (MaNGA) survey which provides us with less contaminated SF and NonSF regions. This makes it an exceptional training set for a range of analytical projects.


In this paper, we introduce a decomposition method, Deep Decomposition Model (DDM), that utilizes the BPT diagram along with deep supervised models to separate galaxy spectra into their constituent NonSF and SF components. This method is refined by training data derived from MaNGA spectra. Training our DDM for spectra decomposition involves generating synthetic spectra with known contributions from nearhomogenous sources. This section outlines how we processed MaNGA spectra to act as these representative sources.


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
